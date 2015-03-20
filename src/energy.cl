//******************************************************************************
//
// LABS energy kernel
//
// With memetics and fireworks.
//
//******************************************************************************




int coleration_for(int k,__global char* agent, int size) {
  int coleration = 0;

  int tail = size - k - 1;
  for (int head = 0;
       head < k;
       ++head) {
    coleration += agent[head] * agent[tail];
    ++tail;
  }

  return coleration * coleration;
}



__kernel void energy( __global char* input,
                      __global int* output,
                      const unsigned int size)
{
  int global_id = get_global_id(0);

  output[global_id] = coleration_for(global_id, input, size);

}
