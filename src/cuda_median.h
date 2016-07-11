#ifndef CUDA_MEDIAN_H
#define CUDA_MEDIAN_H

#ifndef FOLD
#define FOLD if(1)
#endif


/** Calculate the median of an array of float values  .
 *
 * This sorts the actual array so the values will be reordered
 * This uses a bitonicSort which is very fast if the array is in SM
 * This means that there
 *
 * @param array array of floats to search, this will be reordered should be in SM
 * @param arrayLength the number of floats in the array
 * @param dir the direction to sort the array 1 = increasing
 * @return the median value
 */
__device__ float cuMedianOne(float *array, uint arrayLength);


/** Calculate the median of up to 16*bufferSz float values  .
 *
 * This Sorts sections of the array, and then finds the median by extracting and combining the
 * centre chunk(s) of these and sorting those. To find the median.
 *
 * Note this is an in-place selection and reorders the original array
 *
 * @param   data the value to find the median of
 * @param   buffer to do the sorting in this is bufferSz long and should be in SM
 * @param   arrayLength the length of the data array
 * @return  The median of data
 */
 template< int bufferSz >
__device__ float cuMedianBySection(float *data, float *buffer, uint arrayLength);


#endif // CUDA_MEDIAN_H
