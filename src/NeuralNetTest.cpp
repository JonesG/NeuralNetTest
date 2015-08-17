#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;

void Layer1();
void Layer2();
void LayerN();

void main()
{
//	Layer1();
//	Layer2();
	LayerN();
	
	std::cin.ignore();
}

void Layer1()
{
	MatrixXf	X(4, 3);
	ArrayXXf	y(4, 1);

	X <<	0, 0, 1,
			0, 1, 1,
			1, 0, 1,
			1, 1, 1;

	y << 0, 0, 1, 1;

	MatrixXf	syn0 = 2 * (ArrayXXf::Random(3, 1) - 1);
	ArrayXXf	l1;

	for (int j = 0; j < 60000; j++)
	{
		MatrixXf	l0 = X;
					l1 = 1 / (1 + exp(ArrayXXf(-l0*syn0)));
		ArrayXXf	l1_error = y - l1;
		MatrixXf	l1_delta = l1_error - (l1*(1 - l1));

		syn0 += l0.transpose() * l1_delta;
	}

	std::cout << l1 << std::endl;
}

ArrayXXf Deriv(const ArrayXXf & m)
{
	ArrayXXf	a = ArrayXXf(m);

	return a * (1 - a);
}

void Layer2()
{
	MatrixXf	X(5, 3);
	ArrayXXf	y(5, 1);

	X <<	0, 0, 1,
			0, 1, 1,
			1, 0, 1,
			1, 1, 1,
			1, 1, 0;

	y << 0, 1, 1, 0, 0;

	MatrixXf	syn0 = 2 * (ArrayXXf::Random(3, 5) - 1);
	MatrixXf	syn1 = 2 * (ArrayXXf::Random(5, 1) - 1);
	ArrayXXf	l2;

	for (int j = 0; j < 100000; j++)
	{
		MatrixXf	l1 = 1 / (1 + exp(-ArrayXXf(X*syn0)));
					l2 = 1 / (1 + exp(-ArrayXXf(l1*syn1)));
		MatrixXf	l2_delta = (y - l2)*Deriv(l2);
		MatrixXf	l1_delta = ArrayXXf(l2_delta*syn1.transpose()) * Deriv(l1);

		syn1 += l1.transpose()*l2_delta;
		syn0 += X.transpose()*l1_delta;
	}

	std::cout << l2 << std::endl;
}

void LayerN()
{
	// I=inputs, H=hidden layers, V=variables, N=hidden layer elements
	const int	I = 5;
	const int	H = 2;
	const int	V = 3;
	const int	N = 7;

	MatrixXf	L[H+2];
	L[0].resize(I, V);
	L[0] <<	0, 0, 1,
			0, 1, 1,
			1, 0, 1,
			1, 1, 1,
			1, 1, 0;

	ArrayXXf	y(I, 1);
	y << 0, 1, 1, 0, 0;

	MatrixXf	W[H+1];
	
	// first weight layer
	W[0] = 2 * (ArrayXXf::Random(V, N) - 1);
	// weights
	for (int i = 1; i < H; i++)
	{
		W[i] = 2 * (ArrayXXf::Random(N, N) - 1);
	}
	// final weight layer
	W[H] = 2 * (ArrayXXf::Random(N, 1) - 1);

	for (int j = 0; j < 100000; j++)
	{
		// forwards layers
		for (int i = 1; i<H+2; i++)
		{
			L[i] = 1 / (1 + exp(-ArrayXXf(L[i-1] * W[i-1])));
		}

		// backwards output delta
		MatrixXf	LDelta[H+2];
		LDelta[H+1] = (y - ArrayXXf(L[H+1]))*Deriv(L[H+1]);

		// backwards deltas
		for (int i = H; i>0; i--)
		{
			LDelta[i] = ArrayXXf(LDelta[i+1] * W[i].transpose()) * Deriv(L[i]);
		}

		// backwards weights
		for (int i = H; i>=0; i--)
		{
			W[i] += L[i].transpose()*LDelta[i+1];
		}
	}

	std::cout << L[H+1] << std::endl;
}
