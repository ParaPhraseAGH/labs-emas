//******************************************************************************
//
// LABS energy kernel
//
// With memetics and fireworks.
//
//******************************************************************************n
__kernel void reduce( __global char* inputAgent,
                      __global float* output,
                      __global float* bestFitness,
                      __local float* localFitness,
                      __local int* indexes,
                      const unsigned int size)
{
  int localID = get_local_id(0);

  // copy fittness to local memory
  if (localID <= size) {
    localFitness[localID] = bestFitness[localID];
  } else {
    localFitness[localID] = 0;
  }

  // create array of consequtive id's
  indexes[localID] = localID;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset >>= 1) {

    if (localID < offset) {
      // find greates fitness
      float mine = localFitness[localID];
      float other = localFitness[localID + offset];
      localFitness[localID] = (mine > other) ? mine : other;

      // keep track of index of gratest fitness
      int my_index  = indexes[localID];
      int other_index = indexes[localID + offset];
      indexes[localID] = (mine > other) ? my_index : other_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (localID == 0) {
    // return greatest fitness
    output[0] = localFitness[0];

    // mutate inputAgent to corespond to found fitness
    int bit_to_change = indexes[0];

    if (bit_to_change < size) {
      inputAgent[bit_to_change] = (inputAgent[bit_to_change] - 1) * -1;
    }
  }
}
