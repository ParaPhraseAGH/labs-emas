//******************************************************************************
//
// LABS energy kernel
//
// With memetics and fireworks.
//
//******************************************************************************n
__kernel void reduce( __global double* output,
                      __global double* bestFitness,
                      __local double* localFitness,
                      const unsigned int size)
{
  int global_id = get_global_id(0);

  if (global_id <= size) {
    localFitness[global_id] = bestFitness[global_id];
  } else {
    localFitness[global_id] = 0;
  }
   

  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset >>= 1) {
    if (global_id < offset) {
      float other = localFitness[global_id + offset];
      float mine = localFitness[global_id];
      localFitness[global_id] = (mine > other) ? mine : other;
      // TODO save best, and best index
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }


  if (global_id == 0) { //last, not mutated
    output[0] = bestFitness[size];
  }
}
