//******************************************************************************
//
// LABS energy kernel
//
// With memetics and fireworks.
//
//******************************************************************************



int coleration_for(int k,__local char* agent, int size) {
  int coleration = 0;

  int tail = size - k;
  for (int head = 0;
       head < k;
       ++head) {
    coleration += (agent[head] == agent[tail] ? 1 : -1);
    ++tail;
  }

  return coleration;
}



__kernel void energy( __global char* input,
                      __local  char* mutatedAgent,
                      __local int* colerations,
                      __global double* bestFitness,
                      const unsigned int size)
{
  // int global_id = get_global_id(0);


  // copy agent to local memory
  int local_id = get_local_id(0);
  if (local_id < size) {
    mutatedAgent[local_id] = input[local_id];
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  int bitToMutate = get_group_id(0);
  if (bitToMutate == local_id && bitToMutate < size) {
    mutatedAgent[bitToMutate] = (mutatedAgent[bitToMutate] -1) * -1;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (local_id < size) {
    colerations[local_id] = coleration_for(local_id, mutatedAgent, size);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset >>= 1) {
    if (local_id < offset) {
      float other = colerations[local_id + offset];
      float mine = colerations[local_id];
      colerations[local_id] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if(local_id == 0) {
    double energy = colerations[0];
    bestFitness[get_group_id(0)] = size * size * 0.5 / energy;
  }
}
