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
  int localID = get_local_id(0);

  if (localID <= size) {
    localFitness[localID] = bestFitness[localID];
  } else {
    localFitness[localID] = 0;
  }


  indexes[localID] = localID;

  printf("<<< Fit %f for %d\n", localFitness[localID], indexes[localID] );

  barrier(CLK_LOCAL_MEM_FENCE);

  printf("******* Fit %f for %d\n", localFitness[localID], indexes[localID] );

  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset >>= 1) {

    if (localID < offset) {
      double mine = localFitness[localID];
      double other = localFitness[localID + offset];
      localFitness[localID] = (mine > other) ? mine : other;

      if (localID + offset == 32) {
        printf("#### mine: %f other %f\n", mine, other);
      }

      int my_index  = indexes[localID];
      int other_index = indexes[localID + offset];

      
      //printf("my_index: %d other_index %d \n", my_index, other_index);
      indexes[localID] = (mine > other) ? my_index : other_index;
      printf(">>> my_index %d ; other_index %d ; localID %d <<<", my_index, other_index, localID);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (localID == 0) { //last, not mutated
    output[0] = localFitness[0];

    printf("size: %d", size);

    int bit_to_change = indexes[0];

    if (bit_to_change < size) {
      inputAgent[bit_to_change] = (inputAgent[bit_to_change] - 1) * -1;
    }
  }
}
