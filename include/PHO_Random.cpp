#include"PHO_Random.h"

#define RAND48_SEED_0   (0x330e)
#define RAND48_SEED_1   (0xabcd)
#define RAND48_SEED_2   (0x1234)
#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)


GL_DEFINE_SINGLETON(PHO_Random);

void PHO_Random::Init(){
	m_rand48_seed[0] = RAND48_SEED_0;
	m_rand48_seed[1] = RAND48_SEED_1;
	m_rand48_seed[2] = RAND48_SEED_2;

	m_rand48_mult[0] = RAND48_MULT_0;
	m_rand48_mult[1] = RAND48_MULT_1;
	m_rand48_mult[2] = RAND48_MULT_2;

	m_rand48_add = RAND48_ADD;

}
void PHO_Random::SetSeed(unsigned short* Tseed){
	for (int i = 0; i < 3; i++)
		m_rand48_seed[i] = Tseed[i];
}

void PHO_Random::_dorand48(unsigned short xseed[3])
{
	unsigned long accu;
	unsigned short temp[2];

	accu = (unsigned long)m_rand48_mult[0] * (unsigned long)xseed[0] +
		(unsigned long)m_rand48_add;
	temp[0] = (unsigned short)accu;        /* lower 16 bits */
	accu >>= sizeof(unsigned short)* 8;
	accu += (unsigned long)m_rand48_mult[0] * (unsigned long)xseed[1] +
		(unsigned long)m_rand48_mult[1] * (unsigned long)xseed[0];
	temp[1] = (unsigned short)accu;        /* middle 16 bits */
	accu >>= sizeof(unsigned short)* 8;
	accu += m_rand48_mult[0] * xseed[2] + m_rand48_mult[1] * xseed[1] + m_rand48_mult[2] * xseed[0];
	xseed[0] = temp[0];
	xseed[1] = temp[1];
	xseed[2] = (unsigned short)accu;
}

double PHO_Random::erand48(unsigned short xseed[3])
{
	_dorand48(xseed);
	return ldexp((double)xseed[0], -48) +
		ldexp((double)xseed[1], -32) +
		ldexp((double)xseed[2], -16);
}

double PHO_Random::GetDouble(void)
{
	return erand48(m_rand48_seed);
}

long PHO_Random::GetLong(void)
{
	_dorand48(m_rand48_seed);
	return ((long)m_rand48_seed[2] << 15) + ((long)m_rand48_seed[1] >> 1);
}

long PHO_Random::nrand48(unsigned short xseed[3])
{
	_dorand48(xseed);
	return ((long)xseed[2] << 15) + ((long)xseed[1] >> 1);
}

long PHO_Random::mrand48(void)
{
	_dorand48(m_rand48_seed);
	return ((long)m_rand48_seed[2] << 16) + (long)m_rand48_seed[1];
}

long PHO_Random::jrand48(unsigned short xseed[3])
{
	_dorand48(xseed);
	return ((long)xseed[2] << 16) + (long)xseed[1];
}

void PHO_Random::srand48(long seed)
{
	m_rand48_seed[0] = RAND48_SEED_0;
	m_rand48_seed[1] = (unsigned short)seed;
	m_rand48_seed[2] = (unsigned short)(seed >> 16);
	m_rand48_mult[0] = RAND48_MULT_0;
	m_rand48_mult[1] = RAND48_MULT_1;
	m_rand48_mult[2] = RAND48_MULT_2;
	m_rand48_add = RAND48_ADD;
}

unsigned short* PHO_Random::seed48(unsigned short xseed[3])
{
	static unsigned short sseed[3];

	sseed[0] = m_rand48_seed[0];
	sseed[1] = m_rand48_seed[1];
	sseed[2] = m_rand48_seed[2];
	m_rand48_seed[0] = xseed[0];
	m_rand48_seed[1] = xseed[1];
	m_rand48_seed[2] = xseed[2];
	m_rand48_mult[0] = RAND48_MULT_0;
	m_rand48_mult[1] = RAND48_MULT_1;
	m_rand48_mult[2] = RAND48_MULT_2;
	m_rand48_add = RAND48_ADD;
	return sseed;
}

void PHO_Random::lcong48(unsigned short p[7])
{
	m_rand48_seed[0] = p[0];
	m_rand48_seed[1] = p[1];
	m_rand48_seed[2] = p[2];
	m_rand48_mult[0] = p[3];
	m_rand48_mult[1] = p[4];
	m_rand48_mult[2] = p[5];
	m_rand48_add = p[6];
}
