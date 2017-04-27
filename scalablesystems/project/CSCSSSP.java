package scalablesystems.project;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.io.LongWritable;

import in.dream_lab.goffish.api.IEdge;
import in.dream_lab.goffish.api.ISubgraph;
import in.dream_lab.goffish.api.IVertex;


import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.jnvgraph.JNvgraph.nvgraphAllocateEdgeData;
import static jcuda.jnvgraph.JNvgraph.nvgraphAllocateVertexData;
import static jcuda.jnvgraph.JNvgraph.nvgraphConvertTopology;
import static jcuda.jnvgraph.JNvgraph.nvgraphCreate;
import static jcuda.jnvgraph.JNvgraph.nvgraphCreateGraphDescr;
import static jcuda.jnvgraph.JNvgraph.nvgraphDestroy;
import static jcuda.jnvgraph.JNvgraph.nvgraphDestroyGraphDescr;
import static jcuda.jnvgraph.JNvgraph.nvgraphGetVertexData;
import static jcuda.jnvgraph.JNvgraph.nvgraphSssp;
import static jcuda.jnvgraph.JNvgraph.nvgraphSetEdgeData;
import static jcuda.jnvgraph.JNvgraph.nvgraphSetGraphStructure;
import static jcuda.jnvgraph.JNvgraph.nvgraphSetVertexData;
import static jcuda.jnvgraph.nvgraphTopologyType.NVGRAPH_COO_32;
import static jcuda.jnvgraph.nvgraphTopologyType.NVGRAPH_CSC_32;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cudaDataType;
import jcuda.jnvgraph.JNvgraph;
import jcuda.jnvgraph.nvgraphCOOTopology32I;
import jcuda.jnvgraph.nvgraphCSCTopology32I;
import jcuda.jnvgraph.nvgraphGraphDescr;
import jcuda.jnvgraph.nvgraphHandle;
import jcuda.jnvgraph.nvgraphTag;

public class CSCSSSP {
	/*Source to Target Map*/
	private Map<Long, Integer> stotMap;
	/*Target to Source Map*/
	private Map<Integer, Long> ttosMap; 
	int source_indices[];
	int destination_offsets[];
	float CSC_weights[];
	int nedges;
	int nvertices;
	float[] output;
	
	public float getDistance(Long vertexId){
		return output[stotMap.get(vertexId)];
	}
	
	
	public void runSSSP(Long source){
		JNvgraph.setExceptionsEnabled(true);

		/*Should be similar to the JCuda NVgraph example*/
		int targetSource = stotMap.get(source);
		JNvgraph.setExceptionsEnabled(true);
        int  vertex_numsets = 1, edge_numsets = 1;

        // nvgraph variables
        nvgraphHandle handle = new nvgraphHandle();
        nvgraphGraphDescr graph = new nvgraphGraphDescr();
        nvgraphCSCTopology32I CSC_input = new nvgraphCSCTopology32I();
        int edge_dimT = cudaDataType.CUDA_R_32F;
        int vertex_dimT[];

        // Init host data
        output = new float[nvertices];
        vertex_dimT = new int[vertex_numsets];
        vertex_dimT[0] = cudaDataType.CUDA_R_32F; 

        nvgraphCreate(handle);
        nvgraphCreateGraphDescr (handle, graph);

        CSC_input.nvertices = nvertices;
        CSC_input.nedges = nedges;
        CSC_input.destination_offsets = Pointer.to(destination_offsets);
        CSC_input.source_indices = Pointer.to(source_indices);

        // Set graph connectivity and properties (transfers)
        nvgraphSetGraphStructure(
            handle, graph, CSC_input, NVGRAPH_CSC_32);
        nvgraphAllocateVertexData(
            handle, graph, vertex_numsets, Pointer.to(vertex_dimT));
        nvgraphAllocateEdgeData(
            handle, graph, edge_numsets, Pointer.to(new int[] { edge_dimT }));
        nvgraphSetEdgeData(
            handle, graph, Pointer.to(CSC_weights), 0);

        
        nvgraphSssp(handle, graph, 0,  Pointer.to(new int[]{targetSource}), 0);

        nvgraphGetVertexData(handle, graph, Pointer.to(output), 0);

        nvgraphDestroyGraphDescr(handle, graph);
        nvgraphDestroy (handle);
	}
	
	
	public void makeCSCBean(ISubgraph<LongWritable,LongWritable,LongWritable,LongWritable,LongWritable,LongWritable> subGraph){
		JNvgraph.setExceptionsEnabled(true);
		/*Counter keeps track of the temporary vertex IDs*/
		int counter = 0;
		
		/*stotMap holds the mapping from source vertex Id -> target vertex Id
		 * Target vertex Ids always begin from index 0;
		 * */
		stotMap = new HashMap<Long, Integer>();
		ttosMap = new HashMap<Integer, Long>();
		
		List<Integer> sourceIndices = new ArrayList<Integer>();
		List<Integer> destinationIndices = new ArrayList<Integer>();
		List<Float> weights = new ArrayList<Float>();
		
		/*Extract data in COO Format and convert to CSC*/
		for (IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : subGraph.getVertices()) {
			
			/*Remote Vertices are handled later*/
	        if (vertex.isRemote()) {
	          continue;
	        }
	        
	        long sourceVertex = vertex.getVertexId().get();
	        /*If source vertex is not in the source to target Map, establish it's mapping*/
	        if(stotMap.get(sourceVertex) == null){
	        	stotMap.put(sourceVertex, counter);
	        	ttosMap.put(counter, sourceVertex);
	        	counter += 1;
	        }
	        
	        for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
	           long sinkVertex = edge.getSinkVertexId().get();

		       /*If sink vertex is not in the source to target Map, establish it's mapping*/	        	
	           if(stotMap.get(sinkVertex) == null){
	        	   stotMap.put(sinkVertex, counter);
	        	   ttosMap.put(counter, sinkVertex);
	        	   counter += 1;
	           }
	           
	           /*Handle remote vertices by adding the remote -> local edge here. Major assumption is that the edge weights for remote -> local and local -> remote are the same*/
	           if(subGraph.getVertexById(edge.getSinkVertexId()).isRemote()){
	        	   sourceIndices.add(stotMap.get(sinkVertex));
		           destinationIndices.add(stotMap.get(sourceVertex));
		           /*Edge Weights are assumed to be 1*/
		           weights.add(1.0f);
	           }
	           
	           sourceIndices.add(stotMap.get(sourceVertex));
	           destinationIndices.add(stotMap.get(sinkVertex));
	           /*Edge Weights are assumed to be 1*/
	           weights.add(1.0f);
	        } 
	      }
		/*By now the data should be in COO format in the three ArrayLists*/
		/*Number of edges should be the length of any of the three ArrayLists*/
		/*The Number of Vertices should be the length of the keySet of stot or ttos Map or the value of the counter*/
		int source_indices_local[] = new int[sourceIndices.size()];
		for(int i = 0;i < source_indices_local.length;i++)
			source_indices_local[i] = sourceIndices.get(i);
		int destination_indices_local[] = new int[destinationIndices.size()];
		for(int i = 0;i < destination_indices_local.length;i++)
			destination_indices_local[i] = destinationIndices.get(i);
		float weights_local[] = new float[weights.size()];
		for(int i = 0;i < weights_local.length;i++)
			weights_local[i] = weights.get(i);
		
		nedges = source_indices_local.length;
		
		/*TODO: replace with counter*/
		nvertices = stotMap.size();
		
		if(nvertices != counter){
			System.out.println("Something amiss???");
		}
		
		/*Ensure that the number of edges are not more than 2 billion*/
		if(nedges > 2000000000){
			System.out.println("The Subgraph " + subGraph.getSubgraphId().get() + " has more than 2 billion edges");
		}
		
		/*Conversion Begins*/
		// Set up the input topology in COO representation
        nvgraphCOOTopology32I COO_input = new nvgraphCOOTopology32I();
        COO_input.nedges = nedges;
        COO_input.nvertices = nvertices;
        COO_input.tag = nvgraphTag.NVGRAPH_UNSORTED;

        // Allocate memory for the COO representation
        cudaMalloc(COO_input.source_indices, nedges * Sizeof.FLOAT);
        cudaMalloc(COO_input.destination_indices, nedges * Sizeof.FLOAT);
        Pointer coo_weights = new Pointer();
        cudaMalloc(coo_weights, nedges * Sizeof.FLOAT);
        
        // Copy the COO input data from the host to the device 
        cudaMemcpy(COO_input.source_indices, 
            Pointer.to(source_indices_local),
            nedges * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        cudaMemcpy(
            COO_input.destination_indices, 
            Pointer.to(destination_indices_local), 
            nedges * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        cudaMemcpy(coo_weights, 
            Pointer.to(weights_local), 
            nedges * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        
        
        // Prepare the topology in CSC form, which will store the output
        nvgraphCSCTopology32I CSC_output = new nvgraphCSCTopology32I();
        
        // Allocate memory for the CSC representation
        cudaMalloc(CSC_output.source_indices, nedges * Sizeof.FLOAT);
        cudaMalloc(CSC_output.destination_offsets, 
            (nvertices + 1) * Sizeof.FLOAT);
        /*csc_weights will be used as the Edge weight*/
        Pointer csc_weights = new Pointer();
        cudaMalloc(csc_weights, nedges * Sizeof.FLOAT);

        // Create the nvGRAPH handle
        nvgraphHandle handle = new nvgraphHandle();
        nvgraphCreate(handle);
     
        // Execute the conversion
        Pointer dataType = Pointer.to(new int[] { cudaDataType.CUDA_R_32F });
        nvgraphConvertTopology(handle, 
            NVGRAPH_COO_32, COO_input, coo_weights, dataType, 
            NVGRAPH_CSC_32, CSC_output, csc_weights);
		/*Conversion Ends*/
        
        /*Copy data to host*/
        source_indices = new int[nedges];
        cudaMemcpy(Pointer.to(source_indices), 
            CSC_output.source_indices, 
            nedges * Sizeof.INT, cudaMemcpyDeviceToHost);
        
        destination_offsets = new int[nvertices + 1];
        cudaMemcpy(Pointer.to(destination_offsets), 
            CSC_output.destination_offsets, 
            (nvertices + 1) * Sizeof.INT, cudaMemcpyDeviceToHost);
        
        CSC_weights = new float[nedges];
        cudaMemcpy(Pointer.to(CSC_weights), 
            csc_weights, nedges * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        
        /*Copy ends*/
        nvgraphDestroy(handle);
	}
}

