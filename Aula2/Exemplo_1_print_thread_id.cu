#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void meu_kernel( void )
{
	printf( "Meu ID: %d\n" , threadIdx.x );
}

int main( )
{
	// Define a vari�vel de captura de erros
	cudaError_t cudaStatus;

	// Informa o device a ser usado caso exista mais de 1
	cudaStatus = cudaSetDevice( 0 );

	// Testa a fun��o cudaSetDevice retornou erro
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaSetDevice falhou!  Existe dispositivo com suporte a CUDA instalado?" );
		fprintf( stderr , "\n\n%s" , cudaGetErrorString( cudaStatus ) );
		goto Error;
	}

	fprintf( stdout , "Inicio\n" );

	meu_kernel << < 2 , 5 >> > ( );

	// Captura o �ltimo erro ocorrido
	cudaStatus = cudaGetLastError( );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "meu_kernel falhou: %s\n" , cudaGetErrorString( cudaStatus ) );
		goto Error;
	}

	// Sincroniza a execu��o do kernel com a CPU
	cudaStatus = cudaDeviceSynchronize( );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaDeviceSynchronize retornou erro %d ap�s lan�amento do kernel!\n" , cudaStatus );
		goto Error;
	}
	fprintf( stdout , "Fim\n" );
Error:
	// Executa a limpeza GPU
	cudaStatus = cudaDeviceReset( );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaDeviceReset falhou!" );
		return 1;
	}

	return 0;
}