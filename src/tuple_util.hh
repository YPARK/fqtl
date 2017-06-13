#include <tuple>
#include <type_traits>
#include <utility>

#ifndef YPP_TUPLE_UTIL_HH_
#define YPP_TUPLE_UTIL_HH_

////////////////////////////////////////////////////////////////
// apply function, one by one each, to each element of tuple
// e.g.,
// func_apply(
//   [](auto &&x) {
//     std::cout << x.name << std::endl;
//   },
// std::make_tuple(eta_mean1, eta_mean2, eta_mean3, eta_var1));

template <typename Func, typename... Ts>
void func_apply(Func &&func, std::tuple<Ts...> &&tup);

////////////////////////////////////////////////////////////////
// apply function, one by one each, to each element of tuple
// 1. recurse
template <typename Func, typename Tuple, unsigned N>
struct func_apply_impl_t {
  static void run(Func &&f, Tuple &&tup) {
    std::forward<Func>(f)(std::get<N>(std::forward<Tuple>(tup)));
    func_apply_impl_t<Func, Tuple, N - 1>::run(std::forward<Func>(f),
                                               std::forward<Tuple>(tup));
  }
};

// 2. basecase
template <typename Func, typename Tuple>
struct func_apply_impl_t<Func, Tuple, 0> {
  static void run(Func &&f, Tuple &&tup) {
    std::forward<Func>(f)(std::get<0>(std::forward<Tuple>(tup)));
  }
};

template <typename Func, typename... Ts>
void func_apply(Func &&f, std::tuple<Ts...> &&tup) {
  using Tuple = std::tuple<Ts...>;
  func_apply_impl_t<Func, Tuple, sizeof...(Ts) - 1>::run(
      std::forward<Func>(f), std::forward<Tuple>(tup));
}

#endif
