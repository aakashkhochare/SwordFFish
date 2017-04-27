package scalablesystems.project;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Arrays;

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
import static jcuda.jnvgraph.JNvgraph.nvgraphPagerank;
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
import jcuda.runtime.JCuda;



/*Contains the Data in CSC format*/
public class CSCDataBean {
	 /*Source to Target Map*/
	 private Map<Long, Integer> stotMap;
	 /*Target to Source Map*/
	 private Map<Integer, Long> ttosMap; 
	 int source_indices[];
	 int destination_offsets[];
	 float CSC_weights[];
	 /*TODO: Pass the appropriate pageRank*/
	 float pageRank[];
	 int nedges;
	 int nvertices;
	 /*Assumption is that since it's an undirected graph, all the vertices have incoming edges*/
	 float bookmarks[];
	
	 public void setPageRankValueOfVertex(long vertexId, float value){
		this.pageRank[stotMap.get(vertexId)] = value;
	 }
	 public float getPageRankValueOfVertex(long vertexId){
		return this.pageRank[stotMap.get(vertexId).intValue()];
	 }
	
	 public void pageRankGPU(){

		JNvgraph.setExceptionsEnabled(true);
		/*Alpha has to be fixed to 1.0f compulsory*/
		/*TODO: Set a better value of alpha1*/
		float alpha1 = 0.85f;
        int vertex_numsets = 2, edge_numsets = 1;
        Pointer alpha1_p = Pointer.to(new float[]{ alpha1 });
        int i;
        //float pr_1[];
        

        // nvgraph variables
        nvgraphHandle handle = new nvgraphHandle();
        nvgraphGraphDescr graph = new nvgraphGraphDescr();
        nvgraphCSCTopology32I CSC_input;
        int edge_dimT[] = new int[edge_numsets];
        edge_dimT[0] = CUDA_R_32F;
        
        int vertex_dimT[];
 
        for(int tempi=0;tempi<nvertices ; tempi++){
			System.out.println("PageRank just before being Submitted" + pageRank[tempi]);
		}
        
        
        // Allocate host data
        
        Pointer vertex_dim[] = new Pointer[vertex_numsets];
        vertex_dimT = new int[vertex_numsets];
        CSC_input = new nvgraphCSCTopology32I();

        // Initialize host data
        vertex_dim[0] = Pointer.to(bookmarks);
        vertex_dim[1] = Pointer.to(pageRank);
        vertex_dimT[0] = CUDA_R_32F;
        vertex_dimT[1] = CUDA_R_32F;

        // Starting nvgraph
        nvgraphCreate(handle);
        nvgraphCreateGraphDescr(handle, graph);

        CSC_input.nvertices = nvertices;
        CSC_input.nedges = nedges;
        CSC_input.destination_offsets = Pointer.to(destination_offsets);
        CSC_input.source_indices = Pointer.to(source_indices);

        // Set graph connectivity and properties (transfers)
        nvgraphSetGraphStructure(handle, graph, CSC_input, NVGRAPH_CSC_32);
        nvgraphAllocateVertexData(handle, graph, vertex_numsets,
            Pointer.to(vertex_dimT));
        nvgraphAllocateEdgeData(handle, graph, edge_numsets,
            Pointer.to(edge_dimT));
        for (i = 0; i < 2; ++i)
            nvgraphSetVertexData(handle, graph, vertex_dim[i], i);
        nvgraphSetEdgeData(handle, graph, Pointer.to(CSC_weights), 0);


        try{
        	nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 1, 1, 1.0e-6f, 1);
        }catch(Exception e){
        	System.out.println(e);
        }
        // Get and print result
        nvgraphGetVertexData(handle, graph, vertex_dim[1], 1);

        /*TODO: Check this part of the code*/
        for (i = 0; i < nvertices; i++)
            System.out.printf("PageRank of vertex " + ttosMap.get(i) + " is %f\n", pageRank[i]);
        System.out.printf("\n");

        nvgraphDestroyGraphDescr(handle, graph);
        nvgraphDestroy(handle);
	}
	
	
	
	/*All the values are expected to be LongWritable*/
	 public void makeCSCDataBean(ISubgraph<LongWritable,LongWritable,LongWritable,LongWritable,LongWritable,LongWritable> subGraph, long numTotalVertices, 
			List<Long> sourceIndicesRemote,
			List<Long> destinationIndicesRemote,
			List<Float> weightsRemote){
		/*Counter keeps track of the temporary vertex IDs*/
		int counter = 0;
		stotMap = new HashMap<Long, Integer>();
		ttosMap = new HashMap<Integer, Long>();
		
		List<Integer> sourceIndices = new ArrayList<Integer>();
		List<Integer> destinationIndices = new ArrayList<Integer>();
		List<Float> weights = new ArrayList<Float>();
		
		/*TODO: Setup for remote vertices*/
		for(int temp = 0; temp < sourceIndicesRemote.size(); temp++){
			if(stotMap.get(sourceIndicesRemote.get(temp)) == null){
				stotMap.put(sourceIndicesRemote.get(temp), counter);
	        	ttosMap.put(counter, sourceIndicesRemote.get(temp));
	        	counter += 1;
			}
			if(stotMap.get(destinationIndicesRemote.get(temp)) == null){
				stotMap.put(destinationIndicesRemote.get(temp), counter);
	        	ttosMap.put(counter, destinationIndicesRemote.get(temp));
	        	counter += 1;
			}
			
			sourceIndices.add(stotMap.get(sourceIndicesRemote.get(temp)));
			destinationIndices.add(stotMap.get(destinationIndicesRemote.get(temp)));
			weights.add(weightsRemote.get(temp));
		}
		
		/*Extract data in COO Format and convert to CSC*/
		for (IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : subGraph.getVertices()) {
	        if (vertex.isRemote()) {
	          continue;
	        }
	        
	        long sourceVertex = vertex.getVertexId().get();
	        /*If source vertex is not in the source to target Map*/
	        if(stotMap.get(sourceVertex) == null){
	        	stotMap.put(sourceVertex, counter);
	        	ttosMap.put(counter, sourceVertex);
	        	counter += 1;
	        }
	        
	        /*TODO: Checks on null may be required*/
	        
	        /*Never send an island vertex*/
	        long numEdges = 0l;
	        for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
	        	numEdges += 1;
    		}
	        
	        for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
		        /*If sink vertex is not in the source to target Map*/	        	
	           long edgeVertex = edge.getSinkVertexId().get();
	           
	           if(stotMap.get(edgeVertex) == null){
	        	   stotMap.put(edgeVertex, counter);
	        	   ttosMap.put(counter, edgeVertex);
	        	   counter += 1;
	           }
	           
	           /*if(subGraph.getVertexById(edge.getSinkVertexId()).isRemote()){
	        	* Not required anymore
	           }*/
	           
	           /*Expectation is that the source and sink vertices are in the stot and ttos Maps*/
	           sourceIndices.add(stotMap.get(sourceVertex));
	           destinationIndices.add(stotMap.get(edgeVertex));
	           /*TODO: Add a check for numEdges not 0 maybe*/
	           weights.add(1.0f/numEdges);
	        } 
	      }
		JNvgraph.setExceptionsEnabled(true);
		/*By now the data should be in COO format in the three ArrayLists*/
		/*Number of edges should be the length of any of the three ArrayLists*/
		/*The Number of Vertices should be the length of the keySet of stot or ttos Map*/
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
		nvertices = stotMap.size();
		bookmarks = new float[nvertices];
		pageRank  = new float[nvertices];
		float initialValue = 1.0f/numTotalVertices;
		for(int i=0;i<nvertices ; i++){
			bookmarks[i] = 0.0f;
			pageRank[i] = initialValue;
			//System.out.println(pageRank[i]);
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
        cudaMalloc(coo_weights, nedges * 4);
        
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
        


	boolean printResult = false;
        //printResult = true;
        if (printResult)
        {
            System.out.println("nvertices           = " + CSC_output.nvertices);
            System.out.println("nedges              = " + CSC_output.nedges);
            System.out.println("source_indices      = " + 
                Arrays.toString(source_indices));
            System.out.println("destination_offsets = " + 
                Arrays.toString(destination_offsets));
            System.out.println("weights             = " + 
                Arrays.toString(CSC_weights));
        }
        /*Copy ends*/
        //nvgraphDestroy(handle);
	}
}
