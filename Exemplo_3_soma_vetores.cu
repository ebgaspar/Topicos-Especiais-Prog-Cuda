#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addKernel( int* c , const int* a , const int* b )
{
	int i = threadIdx.x;
	c[ i ] = a[ i ] + b[ i ];
}

int main( )
{
	const int arraySize = 5;
	const int a[ arraySize ] = { 1, 2, 3, 4, 5 };
	const int b[ arraySize ] = { 10, 20, 30, 40, 50 };
	int c[ arraySize ] = { 0 };

	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Alocar espaço na memória do device
	cudaStatus = cudaMalloc( ( void** ) &dev_c , arraySize * sizeof( int ) );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaMalloc failed!" );
		goto Error;
	}

	cudaStatus = cudaMalloc( ( void** ) &dev_a , arraySize * sizeof( int ) );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaMalloc failed!" );
		goto Error;
	}

	cudaStatus = cudaMalloc( ( void** ) &dev_b , arraySize * sizeof( int ) );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaMalloc failed!" );
		goto Error;
	}

	// Copia os vetores do host para a device
	cudaStatus = cudaMemcpy( dev_a , a , arraySize * sizeof( int ) , cudaMemcpyHostToDevice );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaMemcpy failed!" );
		goto Error;
	}

	cudaStatus = cudaMemcpy( dev_b , b , arraySize * sizeof( int ) , cudaMemcpyHostToDevice );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaMemcpy failed!" );
		goto Error;
	}

	// Executar o kernel
	addKernel << <1 , arraySize >> > ( dev_c , dev_a , dev_b );

	// Verificar se o kernel foi executado corretamente
	cudaStatus = cudaGetLastError( );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "addKernel launch failed: %s\n" , cudaGetErrorString( cudaStatus ) );
		goto Error;
	}

	// Espera o kernel terminar e retorna quaisquer erros encontrados durante a execução
	cudaStatus = cudaDeviceSynchronize( );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaDeviceSynchronize returned error code %d after launching addKernel!\n" , cudaStatus );
		goto Error;
	}

	// Copia o resultado do device para a memória do host.
	cudaStatus = cudaMemcpy( c , dev_c , arraySize * sizeof( int ) , cudaMemcpyDeviceToHost );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaMemcpy failed!" );
		goto Error;
	}

	printf( "{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n" , c[ 0 ] , c[ 1 ] , c[ 2 ] , c[ 3 ] , c[ 4 ] );

	// Limpa a memória
Error:
	cudaFree( dev_c );
	cudaFree( dev_a );
	cudaFree( dev_b );

	cudaStatus = cudaDeviceReset( );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaDeviceReset failed!" );
		return 1;
	}

	return 0;
}

