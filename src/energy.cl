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
                      __local  char* best,
                      const unsigned int size)
{
  int global_id = get_global_id(0);

  best[global_id] = input[global_id];

  barrier(CLK_LOCAL_MEM_FENCE);

  double energy = 0;

  for (int k = 1; k < size; ++k) {
    int coleration = coleration_for(k, best, size);
    energy +=  coleration * coleration;
  }

  if (global_id == 0 ) {
    output[0] = size * size * 0.5 / energy;
  }

}
