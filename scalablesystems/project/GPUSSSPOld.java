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

public class GPUSSSPOld extends AbstractSubgraphComputation<LongWritable, LongWritable, LongWritable, Text, LongWritable, LongWritable, LongWritable> {
  public GPUSSSPOld() {
  }

  private Map<Long, LongWritable> remoteToSubgraphID = new HashMap<Long, LongWritable>();
  private Map<Long, Float> distanceMap = new HashMap<Long,Float>();
  //private Map<Long, Float> pageRankNew = new HashMap<Long,Float>();
  //private Map<Long, Float> localRemoteEdgeWeights = new HashMap<Long,Float>();
  /*TODO: Set appropriate value according to the graph*/
  private long sourceId = 1;
  private Map<Long, CSCSSSP> cscGraph = new HashMap<Long, CSCSSSP>();

  @Override
  public void compute(Iterable<IMessage<LongWritable, Text>> messages) throws IOException {
	  if(getSuperstep() == 0){
		  
		  for (IRemoteVertex<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> remoteVertex : getSubgraph().getRemoteVertices()) {
	    		remoteToSubgraphID.put(remoteVertex.getVertexId().get(),remoteVertex.getSubgraphId());
		  }
		  
		  boolean flag = false;
		  /*Initialize the distanceMap with FloatMax and look for source vertex*/
		  for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
			  distanceMap.put(vertex.getVertexId().get(), Float.MAX_VALUE);
			  if(vertex.getVertexId().get() == sourceId && !vertex.isRemote()){
				  System.out.println("Source found in subGraph " + getSubgraph().getSubgraphId().get());
				  distanceMap.put(vertex.getVertexId().get(), 0.0f);
				  flag = true;
			  }
		  }
		  /*For every subGraph make the CSCSSSP object*/
		  CSCSSSP cscBean = new CSCSSSP();
		  cscBean.makeCSCBean(getSubgraph());
		  cscGraph.put(getSubgraph().getSubgraphId().get(), cscBean);
		  /*TODO: Source vertex found*/
		  if(flag){
			  System.out.print("Running SSSP on Souce");
			  cscBean.runSSSP(sourceId);
		  
			  /*TODO: If the remote vertices have their distances updated, send them message*/
			  /*Update the distance Map and send Messages*/
			  float toAdd = distanceMap.get(sourceId);
			  for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
				  float vertexDistance = cscBean.getDistance(vertex.getVertexId().get());
				  /*TODO: Put a check on overflow*/
				  if(vertexDistance != Float.MAX_VALUE && (distanceMap.get(vertex.getVertexId().get()) > (vertexDistance + toAdd))){
					  distanceMap.put(vertex.getVertexId().get(), (vertexDistance + toAdd));
					  if(vertex.isRemote()){
						  //prepare to send message
						  String toSend = Long.toString(vertex.getVertexId().get()).concat(",").concat(Float.toString(vertexDistance + toAdd));
						  Text sendMessageText = new Text(toSend);
						  sendMessage(remoteToSubgraphID.get(vertex.getVertexId().get()), sendMessageText);
					  }
				  }
		  		}
		  }
		  voteToHalt();
	  }
	  else{
		  Map<Long, Float> toIterateOn = new HashMap<Long, Float>();
		  Map<Long, Float> remoteVerticesOld = new HashMap<Long, Float>();
		  for (IRemoteVertex<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> remoteVertex : getSubgraph().getRemoteVertices()) {
			  remoteVerticesOld.put(remoteVertex.getVertexId().get(),distanceMap.get(remoteVertex.getVertexId().get()));
		  }
		  /*Iterate over messages and see if SSSP has to be run again*/
		  for (IMessage<LongWritable, Text> message : messages) {
      		String[] messagesToSetup = message.getMessage().toString().split(",");
      		System.out.println("Message Recieved is " + message.getMessage().toString());
      		Long vertexToCheck = Long.parseLong(messagesToSetup[0]);
      		Float valueReceived = Float.parseFloat(messagesToSetup[1]);
      		if( valueReceived < distanceMap.get(vertexToCheck)){
      			if(toIterateOn.get(vertexToCheck) == null){
      				toIterateOn.put(vertexToCheck, valueReceived);
      			}
      			else{
      				if(valueReceived < toIterateOn.get(vertexToCheck)){
      					toIterateOn.put(vertexToCheck, valueReceived);
      				}
      			}
      		}
		  }
      		
      		/*Call SSSP for each vertex in the keySet of toIterateOn*/
      		CSCSSSP currentBean = cscGraph.get(getSubgraph().getSubgraphId().get());
      		for(Long newSouce : toIterateOn.keySet()){
      			System.out.println("Running SSSP for " + newSouce);
      			float toAdd = toIterateOn.get(newSouce);
      			currentBean.runSSSP(newSouce);
      			
      			for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
      			  float vertexDistance = currentBean.getDistance(vertex.getVertexId().get());
      			  /*TODO: Put a check on overflow*/
      			  if(vertexDistance != Float.MAX_VALUE && (distanceMap.get(vertex.getVertexId().get()) > (vertexDistance + toAdd))){
      				  distanceMap.put(vertex.getVertexId().get(), (vertexDistance + toAdd));
      			  }
      			}
      		}
      		
      		/*Iterate on remote vertices to check if their distance has changed if yes, send message*/
      		for (IRemoteVertex<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> remoteVertex : getSubgraph().getRemoteVertices()) {
  			  if(distanceMap.get(remoteVertex.getVertexId().get()) < remoteVerticesOld.get(remoteVertex.getVertexId().get())){
  				String toSend = Long.toString(remoteVertex.getVertexId().get()).concat(",").concat(Float.toString(distanceMap.get(remoteVertex.getVertexId().get())));
				Text sendMessageText = new Text(toSend);
				System.out.println("Sending message " + toSend + " from vertex" + remoteVertex.getVertexId().get());
				sendMessage(remoteVertex.getSubgraphId(), sendMessageText);
  			  }
      		}
		  voteToHalt();
	  }
  }
}  
