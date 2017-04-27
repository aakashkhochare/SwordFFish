package scalablesystems.project;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import in.dream_lab.goffish.api.IEdge;
import in.dream_lab.goffish.api.IMessage;
import in.dream_lab.goffish.api.IRemoteVertex;
import in.dream_lab.goffish.api.IVertex;
import in.dream_lab.goffish.api.AbstractSubgraphComputation;



public class GPUPageRank extends AbstractSubgraphComputation<LongWritable, LongWritable, LongWritable, Text, LongWritable, LongWritable, LongWritable> {



  public GPUPageRank() {



  }

  
  private Map<Long, Float> pageRankOld = new HashMap<Long,Float>();
  private Map<Long, Float> pageRankNew = new HashMap<Long,Float>();
  private Map<Long, Float> localRemoteEdgeWeights = new HashMap<Long,Float>();
  /*TODO: Set appropriate value according to the graph*/
  private long numTotalVertices = 1000000;
  private Map<Long, CSCDataBean> cscGraph = new HashMap<Long, CSCDataBean>();
  @Override
  public void compute(Iterable<IMessage<LongWritable, Text>> messages) throws IOException {
    if (getSuperstep() == 0){
    	/*Iterate over all the vertices find the count and send along with value to the remote vertex*/
    	/*Iterate over the vertices and send the remote vertices their count*/    	
    	Map<Long, LongWritable> remoteToSubgraphID = new HashMap<Long, LongWritable>();
    	Map<LongWritable, String> messagesToSend = new HashMap<LongWritable, String>();
    	
    	for (IRemoteVertex<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> remoteVertex : getSubgraph().getRemoteVertices()) {
    		remoteToSubgraphID.put(remoteVertex.getVertexId().get(),remoteVertex.getSubgraphId());
    	}

    	
    	
    	for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
    		if(vertex.isRemote()) {
  	          continue;
  	        }
    		long count = 0;
    		for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
    			count += 1;
    		}
    		for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
    			if(getSubgraph().getVertexById(edge.getSinkVertexId()).isRemote()){
    				String toSend = Long.toString(vertex.getVertexId().get()).concat(",").concat(Long.toString(edge.getSinkVertexId().get())).concat(",").concat(Float.toString(1.0f/count)).concat(":");
    				LongWritable localSubgraphId = remoteToSubgraphID.get(edge.getSinkVertexId().get());
    				if(messagesToSend.get(localSubgraphId) == null){
    					messagesToSend.put(localSubgraphId, toSend);
    				}
    				else{
    					messagesToSend.put(localSubgraphId, messagesToSend.get(localSubgraphId).concat(toSend));
    				}
    			}
    		}
    	}
    	/*Send the subgraph aggregated messages*/
    	for(LongWritable key : messagesToSend.keySet()){
    		Text sendSubgraphMessage = new Text(messagesToSend.get(key));
    		sendMessage(key, sendSubgraphMessage);
    	}
    }
    else if(getSuperstep() == 1){
    	
    	/*TODO: Set this value up appropriately*/
    	if(getSubgraph().getLocalVertexCount() > 000){
    	
    		/*Set up the remote vertices for the graph*/
        	List<Long> sourceIndices = new ArrayList<Long>();
        	List<Long> destinationIndices = new ArrayList<Long>();
        	List<Float> weights = new ArrayList<Float>();
        	
        	for (IMessage<LongWritable, Text> message : messages) {
        		String[] messagesToSetup = message.getMessage().toString().split(":");
        		System.out.println("Message Recieved is " + message.getMessage().toString());
        		for(int temp = 0; temp < messagesToSetup.length; temp++){
        			String[] parts = messagesToSetup[temp].split(",");
        			sourceIndices.add(Long.parseLong(parts[0]));
        			destinationIndices.add(Long.parseLong(parts[1]));
        			weights.add(Float.parseFloat(parts[2]));
        		}
        	}
        	
    		/*perform GPU Computation*/
    		
    		/*Set up the CSCDataBean*/
    		CSCDataBean currentBean = new CSCDataBean();
    		//long numTotalVertices = 0;

    		currentBean.makeCSCDataBean(getSubgraph(), numTotalVertices, sourceIndices, destinationIndices, weights);

		System.out.println("Bean has been made");
    		cscGraph.put(getSubgraph().getSubgraphId().get(), currentBean);
    		
    		/*Calculate PageRank*/

    		currentBean.pageRankGPU();

    		/*TODO: call the alpha manipulation kernel*/
    		
    		/*Send the computed vertex values to remote vertices*/
    		Map<Long, LongWritable> remoteToSubgraphID = new HashMap<Long, LongWritable>();
        	Map<LongWritable, String> messagesToSend = new HashMap<LongWritable, String>();
        	for (IRemoteVertex<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> remoteVertex : getSubgraph().getRemoteVertices()) {
        		remoteToSubgraphID.put(remoteVertex.getVertexId().get(),remoteVertex.getSubgraphId());
        	}
        	for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
        		if(vertex.isRemote()) {
      	          continue;
      	        }
        		for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
        			if(getSubgraph().getVertexById(edge.getSinkVertexId()).isRemote()){
        				/*Send local vertex's correct pageRank value to the remote subgraph*/
        				String toSend = Long.toString(vertex.getVertexId().get()).concat(",").concat(Float.toString(currentBean.getPageRankValueOfVertex(vertex.getVertexId().get()))).concat(":");
        				LongWritable localSubgraphId = remoteToSubgraphID.get(edge.getSinkVertexId().get());
        				if(messagesToSend.get(localSubgraphId) == null){
        					messagesToSend.put(localSubgraphId, toSend);
        				}
        				else{
        					messagesToSend.put(localSubgraphId, messagesToSend.get(localSubgraphId).concat(toSend));
        				}
        			}
        		}
        	}
        	
    	}
    	else{
    		/*TODO: Write logic to perform the same computation on CPU*/
    		/*Iterate over the Messages and set the remoteVertex edge weights*/
    		for (IMessage<LongWritable, Text> message : messages) {
        		String[] messagesToSetup = message.getMessage().toString().split(":");
        		for(int temp = 0; temp < messagesToSetup.length; temp++){
        			String[] parts = messagesToSetup[temp].split(",");
        			localRemoteEdgeWeights.put(Long.parseLong(parts[0]),Float.parseFloat(parts[2]));
        		}
        	}
    		
    		Map<Long, Float> localPageRankRemote = new HashMap<Long, Float>();
    		
    		/*For all the vertices local and remote init the page rank to 1/totalnumvertices*/
    		for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
    			pageRankOld.put(vertex.getVertexId().get(), 1.0f/numTotalVertices);
    			pageRankNew.put(vertex.getVertexId().get(), 0.0f);
    			if(vertex.isRemote()){
    				localPageRankRemote.put(vertex.getVertexId().get(), 1.0f/numTotalVertices);
    			}	
    		}
    		
    		/*Compute PageRank*/
    		for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
        		if(vertex.isRemote()) {
      	          continue;
      	        }
        		int numberOfEdges = 0;
        		for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
        			numberOfEdges += 1;
        		}
        		/*TODO: Handle remote vertex to local Vertex pageRank*/
        		for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
        			if(getSubgraph().getVertexById(edge.getSinkVertexId()).isRemote()){
        				float currentPageRank = pageRankOld.get(vertex.getVertexId().get()) + localRemoteEdgeWeights.get(edge.getSinkVertexId().get()) * localPageRankRemote.get(edge.getSinkVertexId().get());
        				pageRankNew.put(vertex.getVertexId().get(), currentPageRank);
        			}
        			else{
        				long sinkVertex = edge.getSinkVertexId().get();
        				pageRankNew.put(sinkVertex, pageRankNew.get(sinkVertex) + pageRankOld.get(vertex.getVertexId().get() * numberOfEdges));
        			}
        		}
        	}
    		
    		/*TODO: Multiply pageRank by alpha*/
    		for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
        		if(vertex.isRemote()) {
      	          continue;
      	        }
        		//pageRankOld.put(vertex.getVertexId().get(), 0.85 *); 
        	}
    		/*TODO: Copy pageRankNew into pageRankOld*/
        	/*TODO: Exchange values of remote vertices*/
    	}
    	
    }
    else if(getSuperstep() <= 30){
    	/*Set the values of remote vertices as received from the messages*/
    	if(getSubgraph().getLocalVertexCount() > 00){
    		CSCDataBean currentBean = cscGraph.get(getSubgraph().getSubgraphId().get());
    		for (IMessage<LongWritable, Text> message : messages) {
        		System.out.println("Message Recieved is " + message.toString());
        		String[] messagesToSetup = message.toString().split(":");
        		for(int temp = 0; temp < messagesToSetup.length; temp++){
        			String[] parts = messagesToSetup[temp].split(",");
        			currentBean.setPageRankValueOfVertex(Long.parseLong(parts[0]), Float.parseFloat(parts[1]));
        		}
        	}
    		
    		/*Compute PageRank*/

    		currentBean.pageRankGPU();

    		
    		/*Send the computed vertex values to remote vertices*/
    		Map<Long, LongWritable> remoteToSubgraphID = new HashMap<Long, LongWritable>();
        	Map<LongWritable, String> messagesToSend = new HashMap<LongWritable, String>();
        	for (IRemoteVertex<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> remoteVertex : getSubgraph().getRemoteVertices()) {
        		remoteToSubgraphID.put(remoteVertex.getVertexId().get(),remoteVertex.getSubgraphId());
        	}
        	for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
        		if(vertex.isRemote()) {
      	          continue;
      	        }
        		for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
        			if(getSubgraph().getVertexById(edge.getSinkVertexId()).isRemote()){
        				
        				String toSend = Long.toString(vertex.getVertexId().get()).concat(",").concat(Float.toString(currentBean.getPageRankValueOfVertex(vertex.getVertexId().get()))).concat(":");
        				LongWritable localSubgraphId = remoteToSubgraphID.get(edge.getSinkVertexId().get());
        				if(messagesToSend.get(localSubgraphId) == null){
        					messagesToSend.put(localSubgraphId, toSend);
        				}
        				else{
        					messagesToSend.put(localSubgraphId, messagesToSend.get(localSubgraphId).concat(toSend));
        				}
        			}
        		}
        	}
    		
    		
    	}
    	else{
    		/*TODO: Write logic to perform the same computation on CPU*/
    	}
    }
    else{
    	voteToHalt();
    }
  }
}  
