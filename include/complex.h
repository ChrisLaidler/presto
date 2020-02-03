
typedef long double quad;

///< A templated complex number data type
template<typename T> struct complex_t
{
    T		r;		///< Real Component
    T		i;		///< Imaginary Component

    // default + parameterised constructor
    __host__ __device__ complex_t(T r=0, T i=0)
    : r(r), i(i)
    {
    }

    __host__ __device__ complex_t& operator =(const double v)
    {
      r = v;
      i = v;
      return *this;
    }

    __host__ __device__ complex_t& operator+=(const complex_t& rhs) { r += rhs.r; i += rhs.i; return *this; }
    __host__ __device__ complex_t& operator-=(const complex_t& rhs) { r -= rhs.r; i -= rhs.i; return *this; }

    __host__ __device__ complex_t operator+(const complex_t& a) const
    {
      return complex_t(a.r+r, a.i+i);
    }

    __host__ __device__ complex_t operator-(const complex_t& a) const
    {
      return complex_t(r-a.r, i-a.i);
    }

    __host__ __device__ complex_t operator*(const complex_t& a) const
    {
      return complex_t(r*a.r-i*a.i, r*a.i+i*a.r);
    }

    __host__ __device__ complex_t& operator*=(const complex_t& a)
    {
      T tmpr = r;
      r = r*a.r-i*a.i;
      i = tmpr*a.i+i*a.r;
      return *this;
    }

    __host__ __device__ complex_t operator/(const double v) const
    {
      return complex_t(r/v, i/v);
    }

    __host__ __device__ complex_t operator*(const double v) const
    {
      return complex_t(r*v, i*v);
    }
};

template<typename T> complex_t<T> operator-(const complex_t<T> &f) {return complex_t<T>(-f.r, -f.i);}

template<typename T> complex_t<T> operator*(const float &v, const complex_t<T> &f) {return complex_t<T>(v*f.r, v*f.i);}
template<typename T> complex_t<T> operator/(const float &v, const complex_t<T> &f) {return complex_t<T>(v*f.r/(f.r*f.r+f.i*f.i), -v*f.i/(f.r*f.r+f.i*f.i) );}
template<typename T> complex_t<T> operator-(const float &v, const complex_t<T> &f) {return complex_t<T>(v-f.r, -f.i);}
template<typename T> complex_t<T> operator+(const float &v, const complex_t<T> &f) {return complex_t<T>(v+f.r, f.i);}

template<typename T> complex_t<T> operator*(const double &v, const complex_t<T> &f) {return complex_t<T>(v*f.r, v*f.i);}
template<typename T> complex_t<T> operator/(const double &v, const complex_t<T> &f) {return complex_t<T>(v*f.r/(f.r*f.r+f.i*f.i), -v*f.i/(f.r*f.r+f.i*f.i) );}
template<typename T> complex_t<T> operator-(const double &v, const complex_t<T> &f) {return complex_t<T>(v-f.r, -f.i);}
template<typename T> complex_t<T> operator+(const double &v, const complex_t<T> &f) {return complex_t<T>(v+f.r, f.i);}

template<typename T> complex_t<T> operator*(const quad &v, const complex_t<T> &f) {return complex_t<T>(v*f.r, v*f.i);}
template<typename T> complex_t<T> operator/(const quad &v, const complex_t<T> &f) {return complex_t<T>(v*f.r/(f.r*f.r+f.i*f.i), -v*f.i/(f.r*f.r+f.i*f.i) );}
template<typename T> complex_t<T> operator-(const quad &v, const complex_t<T> &f) {return complex_t<T>(v-f.r, -f.i);}
template<typename T> complex_t<T> operator+(const quad &v, const complex_t<T> &f) {return complex_t<T>(v+f.r, f.i);}


typedef complex_t<float>  fcomplexcu;
typedef complex_t<double> dcomplexcu;
typedef complex_t<quad>   qcomplexcu;

