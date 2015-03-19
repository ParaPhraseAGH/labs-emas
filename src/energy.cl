__kernel void square( __global char* input,
                      __global char* output,
                      const unsigned int count)
{
  int i = get_global_id(0);
  if (i < count)
    output[i] = input[i]*input[i]*i;
}
