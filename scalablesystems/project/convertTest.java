package scalablesystems.project;

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.jnvgraph.JNvgraph.nvgraphAllocateEdgeData;
import static jcuda.jnvgraph.JNvgraph.nvgraphAllocateVertexData;
import static jcuda.jnvgraph.JNvgraph.nvgraphConvertTopology;
import static jcuda.jnvgraph.JNvgraph.nvgraphCreate;
import static jcuda.jnvgraph.JNvgraph.nvgraphCreateGraphDescr;
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

import java.util.Arrays;


import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cudaDataType;
import jcuda.jnvgraph.nvgraphCOOTopology32I;
import jcuda.jnvgraph.nvgraphCSCTopology32I;
import jcuda.jnvgraph.nvgraphGraphDescr;
import jcuda.jnvgraph.nvgraphHandle;
import jcuda.jnvgraph.nvgraphTag;
import jcuda.runtime.JCuda;

/**
 * Basic test for the nvgraphConvertTopology method.
 */
public class convertTest
{
    public static void main(String[] args)
    {
        convertTest t = new convertTest();
        t.testNvgraphConvertTopology();
    }
    
   
    public void testNvgraphConvertTopology()
    {
        JCuda.setExceptionsEnabled(true);
        
        // Set up the values to be copied. This is the topology information
        // given in COO form. The values used here are taken from 
        // http://fareastprogramming.hatenadiary.jp/entry/2016/12/06/203405
        
        /*To be passed to the function as a parameter*/
        float r1 = 1.0f;
        float r2 = 1.0f / 2.0f;
        float r3 = 1.0f / 3.0f;
        int nvertices = 8;
        int nedges = 17;
        long source_indices[] = { 
        		1, 7, 0, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6,0, 6, 7 
        };
        long destination_indices[] = { 
        		3, 6, 1, 1, 4, 1, 4, 5, 5, 6, 7, 7, 0, 4, 2, 7, 5 
        };
        float weights[] = { 
        		r1, r2, r2, r2, r2, r3, r3, r3, r3, r3, r3, r1, r3, r3, r2, r3, r2 
        };

        float bookmark_h[] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        float alpha1 = 0.99899988f;
        Pointer alpha1_p = Pointer.to(new float[]{ alpha1 });
        /*Parameter list ends*/
        
        
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
            Pointer.to(source_indices),
            nedges * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        cudaMemcpy(
            COO_input.destination_indices, 
            Pointer.to(destination_indices), 
            nedges * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        cudaMemcpy(coo_weights, 
            Pointer.to(weights), 
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
        
        
        
        /*Not required. Just a sanity check*/
        // Copy the CSC data from the device to the host
        int host_csc_source_indices[] = new int[nedges];
        cudaMemcpy(Pointer.to(host_csc_source_indices), 
            CSC_output.source_indices, 
            nedges * Sizeof.INT, cudaMemcpyDeviceToHost);
        
        int host_csc_destination_offsets[] = new int[nvertices + 1];
        cudaMemcpy(Pointer.to(host_csc_destination_offsets), 
            CSC_output.destination_offsets, 
            (nvertices + 1) * Sizeof.INT, cudaMemcpyDeviceToHost);
        
        float host_csc_weights[] = new float[nedges];
        cudaMemcpy(Pointer.to(host_csc_weights), 
            csc_weights, nedges * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        
        // TODO: This should not be necessary
        System.out.println("JNvgraphConvertTopologyTest: " + "Setting numbers of vertices and edges in output");
        CSC_output.nvertices = COO_input.nvertices;
        CSC_output.nedges = COO_input.nedges;
        
        
        nvgraphGraphDescr graph = new nvgraphGraphDescr();
        nvgraphCreateGraphDescr(handle, graph);        
        nvgraphSetGraphStructure(handle, graph, CSC_output, NVGRAPH_CSC_32);
        int  vertex_numsets = 3, edge_numsets = 1;
        /*Allocate and set the Vertex Data*/
        int vertex_dimT[] = new int[3];
        vertex_dimT[0] = CUDA_R_32F;
        vertex_dimT[1] = CUDA_R_32F;
        vertex_dimT[2] = CUDA_R_32F;
        
        
        float pr_1[], pr_2[];
        pr_1 = new float[nvertices];
        pr_2 = new float[nvertices];
        Pointer vertex_dim[] = new Pointer[vertex_numsets];
        vertex_dim[0] = Pointer.to(bookmark_h);
        vertex_dim[1] = Pointer.to(pr_1);
        vertex_dim[2] = Pointer.to(pr_2);
        
        nvgraphAllocateVertexData(handle, graph, vertex_numsets,Pointer.to(vertex_dimT));
        for (int i = 0; i < 2; ++i)
            nvgraphSetVertexData(handle, graph, vertex_dim[i], i);
        
        /*Allocate and set the Edge Data*/
        int edge_dimT[] = new int[1];
        edge_dimT[0] = CUDA_R_32F;
        nvgraphAllocateEdgeData(handle, graph, edge_numsets,Pointer.to(edge_dimT));
        nvgraphSetEdgeData(handle, graph, csc_weights, 0);
        
        
        // First run with default values
        try{
        nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 1.0e-6f, 1);
        }catch(Exception e){
        	
        }
        // Get and print result
        nvgraphGetVertexData(handle, graph, vertex_dim[1], 1);
        System.out.printf("pr_1, alpha = 0.85\n");
        for (int i = 0; i < nvertices; i++)
            System.out.printf("%f\n", pr_1[i]);
        System.out.printf("\n");
        
        boolean printResult = false;
        //printResult = true;
        if (printResult)
        {
            System.out.println("nvertices           = " + CSC_output.nvertices);
            System.out.println("nedges              = " + CSC_output.nedges);
            System.out.println("source_indices      = " + 
                Arrays.toString(host_csc_source_indices));
            System.out.println("destination_offsets = " + 
                Arrays.toString(host_csc_destination_offsets));
            System.out.println("weights             = " + 
                Arrays.toString(host_csc_weights));
        }
    }
}        
