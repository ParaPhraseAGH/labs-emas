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

  printf("<<< Fit %f for %d\n", localFitness[local_id], local_id );

  barrier(CLK_GLOBAL_MEM_FENCE);

  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset >>= 1) {
    if (local_id < offset) {
      double mine = localFitness[local_id];
      double other = localFitness[local_id + offset];
      localFitness[local_id] = (mine > other) ? mine : other;

      if (local_id + offset == 35)
      printf("id: %2d mine: %f, other: %f, won: %f \n", local_id, mine, other, localFitness[local_id]);


      double my_index  = indexes[local_id];
      double other_index = indexes[local_id + offset];
      indexes[local_id] = (mine > other) ? my_index : other_index;

    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) { //last, not mutated
    output[0] = localFitness[0];

    //int bit_to_change = indexes[0];



  }
}
