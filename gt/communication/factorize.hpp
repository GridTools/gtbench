#pragma once

#include <type_traits>
#include <array>
#include <algorithm>
#include <limits>

namespace communication {

namespace ghex_comm {

template<std::size_t N, typename I>
void multiply_factor(std::array<I,N>& array, I factor)
{
    I min_element = array[0];
    std::size_t min_index = 0;
    for (std::size_t i=1; i<N; ++i)
    {
        if (array[i] <= min_element)
        {
            min_element = array[i];
            min_index = i;
        }
    }
    array[min_index] *= factor;
}

template<std::size_t N, typename I>
typename std::enable_if<std::is_integral<I>::value, std::array<I,N>>::type
factorize(I n)
{
    std::array<I,N> result;
    result.fill(I{1});
    for (I i=2; i<=n; ++i)
    {
        if (n<2) break;
        if ( (n % i) == 0 )
        {
            multiply_factor(result, i);
            n /= i--;
        }
    }
    std::sort(result.begin(),result.end());
    return result;
}

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

template<std::size_t N, typename I, typename WeightVector>
typename std::enable_if<std::is_integral<I>::value, std::array<I,N>>::type
factorize(I n, const WeightVector& weights)
{
    std::array<I,N> result;
    result.fill(I{1});
    for (I i=2; i<=n; ++i)
    {
        if (n<2) break;
        if ( (n % i) == 0 )
        {
            multiply_factor(result, i, weights);
            n /= i--;
        }
    }
    return result;
}

template<typename I, typename J, std::size_t N>
typename std::enable_if<std::is_integral<I>::value && std::is_integral<J>::value, std::array<std::vector<J>,N>>::type
divide_domain(I n, const std::array<J,N>& sizes)
{
    std::array<std::vector<J>,N> result;
    std::array<double,N> weights;
    for (std::size_t i=0; i<N; ++i) 
        weights[i] = sizes[i];
    const auto factors = factorize<N>(n,weights);
    std::array<double,N> dx;
    for (std::size_t i=0; i<N; ++i) 
    {
        dx[i] = sizes[i]/(double)factors[i];
        while (dx[i]*factors[i] < sizes[i])
        {
            dx[i] += std::numeric_limits<double>::epsilon() * sizes[i];
        }
    }
    
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

