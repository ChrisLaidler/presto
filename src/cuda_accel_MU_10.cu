#include "cuda_accel_MU.h"


__host__  void mult10(cuFFdotBatch* batch)
{
  dim3 dimGrid, dimBlock;

#ifdef SYNCHRONOUS
  cuFfdotStack* pStack = NULL;  // Previous stack
#endif

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  //for (int ss = plains->noStacks-1; ss >= 0; ss-- )
  for (int stack = 0; stack < batch->noStacks; stack++)              // Loop through Stacks
  {
    cuFfdotStack* cStack = &batch->stacks[stack];
    fcomplexcu* d_plainData;    // The complex f-∂f plain data
    fcomplexcu* d_iData;        // The complex input array

    FOLD // Synchronisation  .
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->prepComp,0),   "Waiting for GPU to be ready to copy data to device.");  // Need input data
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->searchComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete

      if      ( (batch->retType & CU_FLOAT) && (batch->retType & CU_STR_PLN) && (batch->flag & FLAG_CUFFT_CB_OUT) )
      {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete
      }
      else if ( (batch->retType & CU_CMPLXF) && (batch->retType & CU_STR_PLN) )
      {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete
      }

#ifdef SYNCHRONOUS
      // Wait for all the input FFT's to complete
      for (int ss = 0; ss< batch->noStacks; ss++)
      {
        cuFfdotStack* cStack2 = &batch->stacks[ss];
        cudaStreamWaitEvent(cStack->multStream, cStack2->prepComp, 0);
      }

      // Wait for the previous multiplication to complete
      if ( pStack != NULL )
        cudaStreamWaitEvent(cStack->multStream, pStack->multComp, 0);
#endif
    }

    FOLD // Timing event  .
    {
#ifdef TIMING
      CUDA_SAFE_CALL(cudaEventRecord(cStack->multInit, cStack->multStream),"Recording event: multInit");
#endif
    }

    FOLD // call kernel(s)  .
    {
      for (int plain = 0; plain < cStack->noInStack; plain++)         // Loop through plains in stack
      {
        cuHarmInfo* cHInfo    = &cStack->harmInf[plain];              // The current harmonic we are working on
        cuFFdot*    cPlain    = &cStack->plains[plain];               // The current f-∂f plain

        dimGrid.x = ceil(cHInfo->width / (float) ( CNV_DIMX * CNV_DIMY ));
        dimGrid.y = 1;

        for (int step = 0; step < batch->noSteps; step++)             // Loop through Steps
        {
          d_iData         = cPlain->d_iData + cStack->strideCmplx * step;

          if      ( batch->flag & FLAG_ITLV_ROW )
          {
            fprintf(stderr,"ERROR: Cannot do single plain multiplications with row-interleaved multi step stacks.\n");
            exit(EXIT_FAILURE);
          }
          else if ( batch->flag & FLAG_ITLV_PLN )
            d_plainData   = cPlain->d_plainData + step * cHInfo->height * cStack->strideCmplx;   // Shift by plain height
          else
            d_plainData   = cPlain->d_plainData;  // If nothing is specified just use plain data

          if ( batch->flag & FLAG_TEX_MUL )
            mult12<<<dimGrid, dimBlock, 0, cStack->multStream>>>(d_plainData, cHInfo->width, cStack->strideCmplx, cHInfo->height, d_iData, cPlain->kernel->kerDatTex);
          else
            mult11<<<dimGrid, dimBlock, 0, cStack->multStream>>>(d_plainData, cHInfo->width, cStack->strideCmplx, cHInfo->height, d_iData, cPlain->kernel->d_kerData);

          // Run message
          CUDA_SAFE_CALL(cudaGetLastError(), "Error at multiplication kernel launch");
        }
      }
    }

    FOLD // Synchronisation  .
    {
      cudaEventRecord(cStack->multComp, cStack->multStream);

#ifdef SYNCHRONOUS
      pStack = cStack;
#endif
    }
  }

}
