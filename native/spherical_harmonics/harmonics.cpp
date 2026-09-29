// Standalone ASSET 0.5.1 extension: no Python callbacks in force evaluation.
#include "VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h"
#include <array>
#include <cmath>

namespace octavian {
// Three-variable Taylor polynomials, with coefficients divided by alpha!.
// Orders 1/2/3 give acceleration/Jacobian/adjoint Hessian, respectively.
constexpr int count(int order) {
  return (order + 1) * (order + 2) * (order + 3) / 6;
}
constexpr auto powers() {
  std::array<std::array<int, 3>, 20> result{};
  int i = 0;
  for (int n = 0; n <= 3; ++n)
    for (int x = n; x >= 0; --x)
      for (int y = n - x; y >= 0; --y)
        result[i++] = {x, y, n - x - y};
  return result;
}
constexpr auto exponents = powers();
constexpr int index(int x, int y, int z) {
  for (int i = 0; i < 20; ++i)
    if (exponents[i][0] == x && exponents[i][1] == y && exponents[i][2] == z)
      return i;
  return -1;
}
constexpr auto products() {
  std::array<std::array<int, 20>, 20> result{};
  for (int i = 0; i < 20; ++i)
    for (int j = 0; j < 20; ++j)
      result[i][j] = index(exponents[i][0] + exponents[j][0],
                           exponents[i][1] + exponents[j][1],
                           exponents[i][2] + exponents[j][2]);
  return result;
}
constexpr auto product_indices = products();
template <int Order> struct Taylor {
  std::array<double, count(Order)> data{};
  Taylor(double value = 0) { data[0] = value; }
  static Taylor variable(double value, int axis) {
    Taylor t(value);
    t.data[axis + 1] = 1;
    return t;
  }
  Taylor operator+(const Taylor &b) const {
    Taylor t;
    for (int i = 0; i < count(Order); ++i)
      t.data[i] = data[i] + b.data[i];
    return t;
  }
  Taylor operator-(const Taylor &b) const {
    Taylor t;
    for (int i = 0; i < count(Order); ++i)
      t.data[i] = data[i] - b.data[i];
    return t;
  }
  Taylor operator*(double b) const {
    Taylor t;
    for (int i = 0; i < count(Order); ++i)
      t.data[i] = data[i] * b;
    return t;
  }
  Taylor operator*(const Taylor &b) const {
    Taylor t;
    for (int i = 0; i < count(Order); ++i)
      for (int j = 0; j < count(Order); ++j) {
        int k = product_indices[i][j];
        if (k >= 0 && k < count(Order))
          t.data[k] += data[i] * b.data[j];
      }
    return t;
  }
  Taylor power(double p) const {
    Taylor u = *this * (1 / data[0]);
    u.data[0] = 0;
    Taylor term(1), result(1);
    double binomial = 1;
    for (int k = 1; k <= Order; ++k) {
      binomial *= (p - k + 1) / k;
      term = term * u;
      result = result + term * binomial;
    }
    return result * std::pow(data[0], p);
  }
  double derivative(int a, int b = -1, int c = -1) const {
    int e[3] = {0, 0, 0};
    ++e[a];
    if (b >= 0)
      ++e[b];
    if (c >= 0)
      ++e[c];
    double f = 1;
    for (int k : e)
      f *= k == 3 ? 6 : (k == 2 ? 2 : 1);
    return data[index(e[0], e[1], e[2])] * f;
  }
};

struct Harmonics : ASSET::VectorFunction<Harmonics, 3, 3> {
  using Base = ASSET::VectorFunction<Harmonics, 3, 3>;
  DENSE_FUNCTION_BASE_TYPES(Base);
  static const bool IsVectorizable = false;
  Eigen::MatrixXd cosine, sine;
  int degree, order;
  Harmonics(Eigen::MatrixXd c, Eigen::MatrixXd s, int n, int m)
      : cosine(std::move(c)), sine(std::move(s)), degree(n), order(m) {
    if (n < 2 || m < 0 || m > n || cosine.rows() <= n ||
        cosine.cols() != cosine.rows() || sine.rows() != cosine.rows() ||
        sine.cols() != cosine.cols() || !cosine.allFinite() ||
        !sine.allFinite())
      throw std::invalid_argument(
          "Invalid spherical-harmonic coefficients or degree/order");
  }
  template <int Order, class In>
  Taylor<Order> potential(const In &input) const {
    using T = Taylor<Order>;
    auto x = T::variable(input[0], 0), y = T::variable(input[1], 1),
         z = T::variable(input[2], 2);
    auto r2 = x * x + y * y + z * z;
    if (!(r2.data[0] > 0) || !std::isfinite(r2.data[0]))
      throw std::invalid_argument("Non-finite or zero harmonic radius");
    auto inv = r2.power(-1), xr = x * inv, yr = y * inv, zr = z * inv;
    auto dv = inv.power(0.5);
    T dw, result;
    for (int m = 0; m <= order; ++m) {
      if (m) {
        double factor =
            m == 1 ? std::sqrt(3.0) : std::sqrt((2.0 * m + 1) / (2 * m));
        auto next = (xr * dv - yr * dw) * factor;
        dw = (xr * dw + yr * dv) * factor;
        dv = next;
      }
      T pv, pw;
      auto v = dv, w = dw;
      for (int n = m; n <= degree; ++n) {
        if (n > m) {
          double a =
              std::sqrt((4.0 * n * n - 1) / (double(n) * n - double(m) * m));
          double b =
              n > m + 1
                  ? std::sqrt((2.0 * n + 1) *
                              (double(n - 1) * (n - 1) - double(m) * m) /
                              ((2.0 * n - 3) * (double(n) * n - double(m) * m)))
                  : 0;
          auto nv = zr * v * a - inv * pv * b, nw = zr * w * a - inv * pw * b;
          pv = v;
          pw = w;
          v = nv;
          w = nw;
        }
        if (n >= 2)
          result = result + v * cosine(n, m) + w * sine(n, m);
      }
    }
    return result;
  }
  template <class In, class Out>
  void compute_impl(ConstVectorBaseRef<In> x,
                    ConstVectorBaseRef<Out> fx_) const {
    auto &fx = fx_.const_cast_derived();
    auto p = potential<1>(x);
    for (int i = 0; i < 3; ++i)
      fx[i] = p.derivative(i);
  }
  template <class In, class Out, class Jac>
  void compute_jacobian_impl(ConstVectorBaseRef<In> x,
                             ConstVectorBaseRef<Out> fx_,
                             ConstMatrixBaseRef<Jac> jx_) const {
    auto &fx = fx_.const_cast_derived();
    auto &jx = jx_.const_cast_derived();
    auto p = potential<2>(x);
    for (int i = 0; i < 3; ++i) {
      fx[i] = p.derivative(i);
      for (int j = 0; j < 3; ++j)
        jx(i, j) = p.derivative(i, j);
    }
  }
  template <class In, class Out, class Jac, class Grad, class Hess, class Adj>
  void compute_jacobian_adjointgradient_adjointhessian_impl(
      ConstVectorBaseRef<In> x, ConstVectorBaseRef<Out> fx_,
      ConstMatrixBaseRef<Jac> jx_, ConstVectorBaseRef<Grad> g_,
      ConstMatrixBaseRef<Hess> h_, ConstVectorBaseRef<Adj> adj) const {
    auto &fx = fx_.const_cast_derived();
    auto &jx = jx_.const_cast_derived();
    auto &g = g_.const_cast_derived();
    auto &h = h_.const_cast_derived();
    auto p = potential<3>(x);
    for (int i = 0; i < 3; ++i) {
      fx[i] = p.derivative(i);
      g[i] = 0;
      for (int j = 0; j < 3; ++j) {
        jx(i, j) = p.derivative(i, j);
        g[i] += jx(i, j) * adj[j];
        h(i, j) = 0;
        for (int k = 0; k < 3; ++k)
          h(i, j) += adj[k] * p.derivative(i, j, k);
      }
    }
  }
};
} // namespace octavian

PYBIND11_MODULE(octavian_harmonics_native, m) {
  auto metadata = py::module_::import("importlib.metadata");
  if (metadata.attr("version")("asset_asrl").cast<std::string>() != "0.5.1")
    throw std::runtime_error(
        "octavian-harmonics-native requires the asset_asrl 0.5.1 pip wheel");
  py::module_::import("asset_asrl");
  if (!py::detail::get_type_info(typeid(ASSET::GenericFunction<-1, -1>), false))
    throw std::runtime_error("ASSET ABI mismatch: build with the pinned ASSET "
                             "headers and matching Clang/pybind11 settings");
  m.def("acceleration_function",
        [](Eigen::MatrixXd c, Eigen::MatrixXd s, int degree, int order) {
          return ASSET::GenericFunction<-1, -1>(
              octavian::Harmonics(std::move(c), std::move(s), degree, order));
        });
  m.attr("asset_version") = "0.5.1";
}
