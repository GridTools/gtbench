#pragma once

#include <type_traits>
#include <array>
#include <algorithm>
#include <limits>

namespace communication {

namespace ghex_comm {

// multiply one element of an array with a factor
// the element e_i with the largest weight_i/e_i ratio is selected 
template<std::size_t N, typename I, typename WeightVector>
void multiply_factor(std::array<I,N>& array, I factor, const WeightVector& weights)
{
    auto max_weight = weights[0]/array[0];
    std::size_t max_index = 0;
    for (std::size_t i=1; i<N; ++i)
    {
        if (weights[i]/array[i] >= max_weight)
        {
            max_weight = weights[i]/array[i];
            max_index = i;
        }
    }
    array[max_index] *= factor;
}

// factorize a number n into N factors f_i
// the factors f_i are chosen so that the ratio weight_i/f_i are balanced
template<std::size_t N, typename I, typename WeightVector>
typename std::enable_if<std::is_integral<I>::value, std::array<I,N>>::type
factorize(I n, const WeightVector& weights)
{
    std::array<I,N> result;
    result.fill(I{1});
    for (I i=2; i<=n; ++i)
    {
        if ( (n % i) == 0 )
        {
            multiply_factor(result, i, weights);
            n /= i--;
        }
    }
    return result;
}

// given an N-dimensional domain with extent sizes:
// split the domain in n sub-domains, and keep their shape
// close to a (hyper-)cube.
template<typename I, typename J, std::size_t N>
typename std::enable_if<std::is_integral<I>::value && std::is_integral<J>::value, std::array<std::vector<J>,N>>::type
divide_domain(I n, const std::array<J,N>& sizes)
{
    // return an array of vectors of sub-extents
    // the vectors contain the sizes of the subdomains along one dimension
    std::array<std::vector<J>,N> result;
    // the weights are equal to the extents
    std::array<double,N> weights;
    for (std::size_t i=0; i<N; ++i) 
        weights[i] = sizes[i];
    // factorize the N-dimensional domain into  n_1 x n_2 x .. x n_N sub-domain factors
    const auto factors = factorize<N>(n,weights);
    // compute the extents dx_i of the sub-domains (as rational numbers)
    std::array<double,N> dx;
    for (std::size_t i=0; i<N; ++i) 
    {
        dx[i] = sizes[i]/(double)factors[i];
        // make sure we get the correct total extent in the end
        while (dx[i]*factors[i] < sizes[i])
        {
            dx[i] += std::numeric_limits<double>::epsilon() * sizes[i];
        }
    }
    // round the dx to integers
    for (std::size_t i=0; i<N; ++i) 
    {
        result[i].resize(factors[i],1);
        for (I j=0; j<factors[i]; ++j)
        {
            const I l = j*dx[i];
            const I u = (j+1)*dx[i];
            result[i][j] = u-l;
        }
        std::sort(result[i].begin(), result[i].end());
    }
    return result;
}


} // namespace ghex_comm

} // namespace communication

