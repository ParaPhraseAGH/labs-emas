//******************************************************************************
//
// LABS energy kernel
//
// With memetics and fireworks.
//
//******************************************************************************n
__kernel void reduce( __global char* inputAgent,
                      __global double* output,
                      __global double* bestFitness,
                      __local double* localFitness,
                      __local int* indexes,
                      const unsigned int size)
{
  int local_id = get_local_id(0);

  if (local_id <= size) {
    localFitness[local_id] = bestFitness[local_id];
  } else {
    localFitness[local_id] = 0;
  }

  indexes[local_id] = local_id;

  //printf("<<< Fit %f for %d\n", localFitness[local_id], indexes[local_id] );

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset >>= 1) {
    if (local_id < offset) {
      double mine = localFitness[local_id];
      double other = localFitness[local_id + offset];
      localFitness[local_id] = (mine > other) ? mine : other;

      int my_index  = indexes[local_id];
      int other_index = indexes[local_id + offset];

      //printf("my_index: %d other_index %d \n", my_index, other_index);
      indexes[local_id] = (mine > other) ? my_index : other_index;
      printf(">> my_index %d ; other_index %d ; local_id %d <<", my_index, other_index, local_id);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) { //last, not mutated
    output[0] = localFitness[0];

    int bit_to_change = indexes[0];
    printf("mutating bin %d\n", bit_to_change);
    printf("!!!!! local_size %d !!!!!", get_local_size(0));

    if (bit_to_change < size) {
      inputAgent[bit_to_change] = (inputAgent[bit_to_change] - 1) * -1;
    }
  }
}
