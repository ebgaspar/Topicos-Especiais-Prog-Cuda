#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
int main( int argc , char** argv )
{

	fprintf( stdout , "CUDA Device Query\n" );

	int deviceCount = 0;

	// Testa se existem dispositivos compat�veis com Cuda
	cudaError_t cudaStatus = cudaGetDeviceCount( &deviceCount );

	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaGetDeviceCount retornou c�digo: %d\n -> %s\n" , cudaStatus , cudaGetErrorString( cudaStatus ) );
		exit( 1 );
	}

	// A fun��o retorna 0 caso n�o exista hardware que suporte cuda.
	if ( deviceCount == 0 )
	{
		fprintf( stdout , "N�o h� dispositivo compat�vel com CUDA\n" );
	}
	else
	{
		fprintf( stdout , "Detectado %d dispositivo(s) CUDA\n" , deviceCount );
	}

	int dev , driverVersion = 0 , runtimeVersion = 0;

	for ( dev = 0; dev < deviceCount; ++dev )
	{
		cudaSetDevice( dev );
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties( &deviceProp , dev );

		fprintf( stdout , "\nDevice %d: \"%s\"\n" , dev , deviceProp.name );

		cudaDriverGetVersion( &driverVersion );
		cudaRuntimeGetVersion( &runtimeVersion );
		cudaDriverGetVersion( &driverVersion );
		cudaRuntimeGetVersion( &runtimeVersion );
		fprintf( stdout , "CUDA Driver Version / Runtime Version %d.%d / %d.%d\n" , driverVersion / 1000 , ( driverVersion % 100 ) / 10 , runtimeVersion / 1000 , ( runtimeVersion % 100 ) / 10 );
		fprintf( stdout , "CUDA Capability Major/Minor version number: %d.%d\n" , deviceProp.major , deviceProp.minor );
		fprintf( stdout , "QTD Multiprocessors: %d \n" , deviceProp.multiProcessorCount );
		fprintf( stdout , "Total constant memory:%zu bytes\n" , deviceProp.totalConstMem );
		fprintf( stdout , "Total shared memory per block:%zu bytes\n" , deviceProp.sharedMemPerBlock );
		fprintf( stdout , "Shared memory per multiprocessor:%zu bytes\n" , deviceProp.sharedMemPerMultiprocessor );
		fprintf( stdout , "Number of registers available per block:%d\n" , deviceProp.regsPerBlock );
	}

	return 0;
}