#include "cuda_accel_MU.h"


__host__  void mult10(cuFFdotBatch* batch)
{
  dim3 dimGrid, dimBlock;

#ifdef SYNCHRONOUS
  cuFfdotStack* pStack = NULL;  // Previous stack
#endif

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  //for (int ss = planes->noStacks-1; ss >= 0; ss-- )
  for (int stack = 0; stack < batch->noStacks; stack++)              // Loop through Stacks
  {
    cuFfdotStack* cStack = &batch->stacks[stack];
    fcomplexcu* d_planeData;    // The complex f-∂f plane data
    fcomplexcu* d_iData;        // The complex input array

    FOLD // Synchronisation  .
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->prepComp,0),     "Waiting for GPU to be ready to copy data to device.");  // Need input data

      // CFF output callback has its own data so can start once FFT is complete
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

//      // Now always using a powers plane
//      if ( (batch->flag & FLAG_CUFFT_CB_OUT) )
//      {
//        // CFF output callback has its own data so can start once FFT is complete
//        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
//      }
//      else
//      {
//        // Have to wait for search to finish reading data
//        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
//      }

      if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
      {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp, 0), "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
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
      for (int plane = 0; plane < cStack->noInStack; plane++)         // Loop through planes in stack
      {
        cuHarmInfo* cHInfo    = &cStack->harmInf[plane];              // The current harmonic we are working on
        cuFFdot*    cPlane    = &cStack->planes[plane];               // The current f-∂f plane

        dimGrid.x = ceil(cHInfo->width / (float) ( CNV_DIMX * CNV_DIMY ));
        dimGrid.y = 1;

        for (int step = 0; step < batch->noSteps; step++)             // Loop through Steps
        {
          d_iData         = cPlane->d_iData + cStack->strideCmplx * step;

          if      ( batch->flags & FLAG_ITLV_ROW )
          {
            fprintf(stderr,"ERROR: Cannot do single plane multiplications with row-interleaved multi step stacks.\n");
            exit(EXIT_FAILURE);
          }
          else
            d_planeData   = cPlane->d_planeMult + step * cHInfo->height * cStack->strideCmplx;   // Shift by plane height

					// Texture memory in multiplication is now beprecated
          //if ( batch->flag & FLAG_TEX_MUL )
          //  mult12<<<dimGrid, dimBlock, 0, cStack->multStream>>>(d_planeData, cHInfo->width, cStack->strideCmplx, cHInfo->height, d_iData, cPlane->kernel->kerDatTex);
          //else
          mult11<<<dimGrid, dimBlock, 0, cStack->multStream>>>(d_planeData, cHInfo->width, cStack->strideCmplx, cHInfo->height, d_iData, cPlane->kernel->d_kerData);

          // Run message
          CUDA_SAFE_CALL(cudaGetLastError(), "At multiplication kernel launch");
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
