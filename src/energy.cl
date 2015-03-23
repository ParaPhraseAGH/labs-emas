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
                      __global double* output,
                      __local  char* bestAgent,
                      __local char* scratch, // N*(N + 1) size;
                      __local float* bestFittnes,
                      const unsigned int size)
{
  int global_id = get_global_id(0);

  if (global_id < size) {
    bestAgent[global_id] = input[global_id];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // TODO create modyfied copy in scratch

  double energy = 0;

  for (int k = 1; k < size; ++k) {
    int coleration = coleration_for(k, bestAgent, size);
    energy +=  coleration * coleration;
  }

  // TODO assign to local best
  if (global_id == 0 ) {
    output[0] = size * size * 0.5 / energy;
  }
  /*
  barrier(CLK_LOCAL_MEM_FENCE);


  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset >>= 1) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = (mine < other) ? mine : other;
      // TODO save best, and best index
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }


  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }

  if (local_index == 0) {
    // TODO return best
    result[get_group_id(0)] = scratch[0];
  }

  // ??? Two stage reduction for greater problems ???
  */

}
