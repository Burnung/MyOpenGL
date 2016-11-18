#ifndef PHO_RANDOM_H
#define PHO_RANDOM_H

#include <math.h>
#include <stdlib.h>

class PHO_Random{

public:
	PHO_Random();
	~PHO_Random(){};
	void SetSeed(unsigned short *);

private:
	unsigned short m_rand48_seed[3];
	unsigned short _rand48_mult[3];
	unsigned short _rand48_add;

	double drand48();

private:

	void _dorand48(unsigned short xseed[3]);


	double erand48(unsigned short xseed[3]);
	



	long lrand48(void);

	long nrand48(unsigned short xseed[3]);

	long mrand48(void);

	long jrand48(unsigned short xseed[3]);


	void srand48(long seed);


	unsigned short * seed48(unsigned short xseed[3]);

	void lcong48(unsigned short p[7]);


};




#endif