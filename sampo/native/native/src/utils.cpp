#include "native/utils.h"

// TODO Think about make own more cache-friendly shuffle using areas around
// target element
vector<int> sample_ind(int n, float prob, random_device &rd) {
    vector<int> result;
    result.resize(n);
    std::iota(result.begin(), result.end(), 0);
    std::shuffle(result.begin(), result.end(), rd);

    result.resize((int)(prob * (float)n));
    return result;
}

int randInt(int min, int max) {
    // TODO implement using C++ 11 and without re-creating distribution
    // objects each call
    std::random_device rd;    // Only used once to initialise (seed) engine
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(min, max);
    return uni(rng);
}

//    /**
//     * Argsort(currently support ascending sort)
//     * @tparam T array element type
//     * @param array input array
//     * @return indices w.r.t sorted array
//     */
//    template<typename T>
//    std::vector<size_t> argsort(const std::vector<T> &array) {
//        std::vector<size_t> indices(array.size());
//        std::iota(indices.begin(), indices.end(), 0);
//        std::sort(indices.begin(), indices.end(),
//                  [&array](int left, int right) -> bool {
//                      // sort indices according to corresponding array
//                      element return array[left] < array[right];
//                  });
//
//        return indices;
//    }

/**
 * Argsort(currently support ascending sort)
 * @tparam T array element type(specified Chromosome*)
 * @param array input array
 * @return indices w.r.t sorted array
 */
std::vector<size_t> argsort(const std::vector<Chromosome *> &array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(
            indices.begin(),
            indices.end(),
            [&array](int left, int right) -> bool {
                // sort indices according to corresponding array element
                return array[left]->fitness < array[right]->fitness;
            }
    );

    //        for (int i = 0; i < indices.size(); i++) {
    //            cout << array[indices[i]]->fitness << " ";
    //        }
    //        cout << endl;

    return indices;
}
