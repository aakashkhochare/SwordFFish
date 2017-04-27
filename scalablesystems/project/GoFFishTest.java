package scalablesystems.project;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.LongWritable;

import in.dream_lab.goffish.api.AbstractSubgraphComputation;
import in.dream_lab.goffish.api.IMessage;
import in.dream_lab.goffish.api.IRemoteVertex;
import in.dream_lab.goffish.api.ISubgraph;
import in.dream_lab.goffish.api.IVertex;

public class GoFFishTest extends AbstractSubgraphComputation<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> {

	  @Override
	  public void compute(Iterable<IMessage<LongWritable, LongWritable>> messages) throws IOException {
		  if(getSuperstep() == 0){
			  ISubgraph<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> subGraph = getSubgraph();
			  long size = 0;
			  if(subGraph.getRemoteVertices() != null){
				Iterator<IRemoteVertex<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable>> itr = subGraph.getRemoteVertices().iterator();
			  	while(itr.hasNext()){
			  		  size += 1;
			  		  itr.next();
			  		  
			  	}
			  	
			  	//if(getSubgraph().getLocalVertexCount() > 40000l){
			  	//System.out.println("For SubGraphID " + subGraph.getSubgraphId() + " the number of Vertices " + subGraph.getVertexCount() + " and the number of local vertices " + subGraph.getLocalVertexCount() + " Number of Remote Vertices is  " + size);
			  	
			  }	
			  else{
				  	//System.out.println("For SubGraphID " + subGraph.getSubgraphId() + " the number of Vertices " + subGraph.getVertexCount() + " and the number of local vertices " + subGraph.getLocalVertexCount() + " Number of Remote Vertices is 0");
			  }
			  
			  for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()){
				  if(vertex.getVertexId().get() == 1l){
					  System.out.print("Vertex found in " + subGraph.getSubgraphId().get());
				  }
			  }
		  }
		  else
			  voteToHalt();
	   }
}
