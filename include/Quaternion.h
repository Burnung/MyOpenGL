#pragma once

/*#include <Constructs/Vec3f.h>
#include <Constructs/Vec4f.h>
#include <Constructs/Matrix4x4f.h>*/
#include<iostream>
#include<glm/vec3.hpp>
//#include"Matrix4x4f.h"
#include<glm/vec4.hpp>
#include<glm/geometric.hpp>
#include <math.h>

using namespace glm;

const float pif = 3.1415926535898;


class Quaternion
{
public:
	static const float s_quaternionNormalizationTolerance;
	static const float s_quaternionDotTolerance;

	float w, x, y, z;

	Quaternion();
	Quaternion(float W, float X, float Y, float Z);
	Quaternion(const vec3 &axis, float angle);
	Quaternion(const vec3 &eulerAngles);

	float Magnitude() const;
	float MagnitudeSquared() const;
	Quaternion Normalize() const;
	void NormalizeThis();

	vec4 GetVec4f() const;

	float Dot(const Quaternion &other) const;

	Quaternion operator+(const Quaternion &other) const;
	Quaternion operator-(const Quaternion &other) const;
	
	Quaternion operator*(float scalar) const;
	Quaternion operator/(float scalar) const;

	//Quaternion operator*(const Quaternion &other) const;

	const Quaternion &operator+=(const Quaternion &other);
	const Quaternion &operator-=(const Quaternion &other);

	const Quaternion &operator*=(const Quaternion &other);

	const Quaternion &operator*=(float scalar);
	const Quaternion &operator/=(float scalar);

	void Rotate(float angle, const vec3 &axis);

	static Quaternion GetRotated(float angle, const vec3 &axis);

	Quaternion Conjugate() const;
	Quaternion Inverse() const;

	void Reset();

	//Matrix4x4f GetMatrix() const;

	void SetFromEulerAngles(const vec3 &eulerAngles);
	vec3 GetEulerAngles() const;

	void CalculateWFromXYZ();

	void MulMatrix();

	static Quaternion Lerp(const Quaternion &first, const Quaternion &second, float interpolationCoefficient);
	static Quaternion Slerp(const Quaternion &first, const Quaternion &second, float interpolationCoefficient);
};

//Vec3f operator*(const Quaternion &quat, const Vec3f &vec);
//Vec3f operator*(const Vec3f &vec, const Quaternion &quat);

Quaternion operator*(const Quaternion& q, const vec3& v);
Quaternion operator*(const Quaternion& q, const Quaternion& v);
std::ostream &operator<<(std::ostream &output, const Quaternion &quat);