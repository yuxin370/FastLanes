// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/include/generated-bindings/kernel-bindings.cuh
// ────────────────────────────────────────────────────────
#ifndef GENERATED_KERNEL_BINDINGS_CUH
#define GENERATED_KERNEL_BINDINGS_CUH

#include "engine/enums.cuh"
#include "decompression/alp.cuh"
#include <cstdint>

namespace galp::bench::bindings {

template <typename T, typename ColumnT>
T* decompress_column(const ColumnT         column,
                     const unsigned        unpack_n_vectors,
                     const unsigned        unpack_n_values,
                     const galp::format::Unpacker unpacker,
                     const galp::format::Patcher  patcher,
                     const galp::format::Expander expander,
                     const uint32_t        n_samples);

template <typename T, typename ColumnT>
T* decompress_column(const ColumnT         column,
                     const unsigned        unpack_n_vectors,
                     const unsigned        unpack_n_values,
                     const galp::format::Unpacker unpacker,
                     const galp::format::Patcher  patcher,
                     const galp::format::Expander expander,
                     const uint32_t        n_samples,
                     const bool            use_shuffle);

template <typename T, typename ColumnT>
bool query_column(const ColumnT         column,
                  const unsigned        unpack_n_vectors,
                  const unsigned        unpack_n_values,
                  const galp::format::Unpacker unpacker,
                  const galp::format::Patcher  patcher,
                  const T               magic_value,
                  const uint32_t        n_samples);

template <typename T, typename ColumnT>
bool compute_column(const ColumnT         column,
                    const unsigned        unpack_n_vectors,
                    const unsigned        unpack_n_values,
                    const galp::format::Unpacker unpacker,
                    const galp::format::Patcher  patcher,
                    const unsigned        n_repetitions,
                    const uint32_t        n_samples);

template <typename T, typename ColumnT>
bool query_multi_column(const ColumnT&        column,
                        const unsigned        unpack_n_vectors,
                        const unsigned        unpack_n_values,
                        const galp::format::Unpacker unpacker,
                        const galp::format::Patcher  patcher,
                        const T               magic_value,
                        const uint32_t        n_samples);

uint32_t* decompress_column(const galp::codec::device::BPColumn<uint32_t> column,
                            const unsigned                           unpack_n_vectors,
                            const unsigned                           unpack_n_values,
                            const galp::format::Unpacker                    unpacker,
                            const galp::format::Patcher                     patcher,
                            const galp::format::Expander                    expander,
                            const uint32_t                           n_samples);
uint64_t* decompress_column(const galp::codec::device::BPColumn<uint64_t> column,
                            const unsigned                           unpack_n_vectors,
                            const unsigned                           unpack_n_values,
                            const galp::format::Unpacker                    unpacker,
                            const galp::format::Patcher                     patcher,
                            const galp::format::Expander                    expander,
                            const uint32_t                           n_samples);
uint32_t* decompress_column(const galp::codec::device::FFORColumn<uint32_t> column,
                            const unsigned                             unpack_n_vectors,
                            const unsigned                             unpack_n_values,
                            const galp::format::Unpacker                      unpacker,
                            const galp::format::Patcher                       patcher,
                            const galp::format::Expander                      expander,
                            const uint32_t                             n_samples);
uint64_t* decompress_column(const galp::codec::device::FFORColumn<uint64_t> column,
                            const unsigned                             unpack_n_vectors,
                            const unsigned                             unpack_n_values,
                            const galp::format::Unpacker                      unpacker,
                            const galp::format::Patcher                       patcher,
                            const galp::format::Expander                      expander,
                            const uint32_t                             n_samples);
uint32_t* decompress_column(const galp::codec::device::SLPATCHColumn<uint32_t> column,
                            const unsigned                                unpack_n_vectors,
                            const unsigned                                unpack_n_values,
                            const galp::format::Unpacker                         unpacker,
                            const galp::format::Patcher                          patcher,
                            const galp::format::Expander                         expander,
                            const uint32_t                                n_samples);
uint64_t* decompress_column(const galp::codec::device::SLPATCHColumn<uint64_t> column,
                            const unsigned                                unpack_n_vectors,
                            const unsigned                                unpack_n_values,
                            const galp::format::Unpacker                         unpacker,
                            const galp::format::Patcher                          patcher,
                            const galp::format::Expander                         expander,
                            const uint32_t                                n_samples);
int16_t*  decompress_column(const galp::codec::device::SLPATCHColumn<int16_t> column,
                            const unsigned                               unpack_n_vectors,
                            const unsigned                               unpack_n_values,
                            const galp::format::Unpacker                        unpacker,
                            const galp::format::Patcher                         patcher,
                            const galp::format::Expander                        expander,
                            const uint32_t                               n_samples);
uint32_t* decompress_column(const galp::codec::device::DICTSLPATCHColumn<uint32_t> column,
                            const unsigned                                    unpack_n_vectors,
                            const unsigned                                    unpack_n_values,
                            const galp::format::Unpacker                             unpacker,
                            const galp::format::Patcher                              patcher,
                            const galp::format::Expander                             expander,
                            const uint32_t                                    n_samples);
uint64_t* decompress_column(const galp::codec::device::DICTSLPATCHColumn<uint64_t> column,
                            const unsigned                                    unpack_n_vectors,
                            const unsigned                                    unpack_n_values,
                            const galp::format::Unpacker                             unpacker,
                            const galp::format::Patcher                              patcher,
                            const galp::format::Expander                             expander,
                            const uint32_t                                    n_samples);
uint32_t* decompress_column(const galp::codec::device::RLEColumn<uint32_t, uint32_t> column,
                            const unsigned                                      unpack_n_vectors,
                            const unsigned                                      unpack_n_values,
                            const galp::format::Unpacker                               unpacker,
                            const galp::format::Patcher                                patcher,
                            const galp::format::Expander                               expander,
                            const uint32_t                                      n_samples);
uint64_t* decompress_column(const galp::codec::device::RLEColumn<uint64_t, uint64_t> column,
                            const unsigned                                      unpack_n_vectors,
                            const unsigned                                      unpack_n_values,
                            const galp::format::Unpacker                               unpacker,
                            const galp::format::Patcher                                patcher,
                            const galp::format::Expander                               expander,
                            const uint32_t                                      n_samples);
uint32_t* decompress_column(const galp::codec::device::CONSTANTColumn<uint32_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const galp::format::Unpacker                          unpacker,
                            const galp::format::Patcher                           patcher,
                            const galp::format::Expander                          expander,
                            const uint32_t                                 n_samples);
uint64_t* decompress_column(const galp::codec::device::CONSTANTColumn<uint64_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const galp::format::Unpacker                          unpacker,
                            const galp::format::Patcher                           patcher,
                            const galp::format::Expander                          expander,
                            const uint32_t                                 n_samples);
uint32_t* decompress_column(const galp::codec::device::DICTFFORColumn<uint32_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const galp::format::Unpacker                          unpacker,
                            const galp::format::Patcher                           patcher,
                            const galp::format::Expander                          expander,
                            const uint32_t                                 n_samples,
                            const bool                                     use_shuffle);
uint64_t* decompress_column(const galp::codec::device::DICTFFORColumn<uint64_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const galp::format::Unpacker                          unpacker,
                            const galp::format::Patcher                           patcher,
                            const galp::format::Expander                          expander,
                            const uint32_t                                 n_samples,
                            const bool                                     use_shuffle);
uint32_t* decompress_column(const galp::codec::device::CROSSRLEExtendedColumn<uint32_t> column,
                            const unsigned                                         unpack_n_vectors,
                            const unsigned                                         unpack_n_values,
                            const galp::format::Unpacker                                  unpacker,
                            const galp::format::Patcher                                   patcher,
                            const galp::format::Expander                                  expander,
                            const uint32_t                                         n_samples);
uint64_t* decompress_column(const galp::codec::device::CROSSRLEExtendedColumn<uint64_t> column,
                            const unsigned                                         unpack_n_vectors,
                            const unsigned                                         unpack_n_values,
                            const galp::format::Unpacker                                  unpacker,
                            const galp::format::Patcher                                   patcher,
                            const galp::format::Expander                                  expander,
                            const uint32_t                                         n_samples);
uint32_t* decompress_column(const galp::codec::device::CROSSRLELaneMaskColumn<uint32_t> column,
                            const unsigned                                         unpack_n_vectors,
                            const unsigned                                         unpack_n_values,
                            const galp::format::Unpacker                                  unpacker,
                            const galp::format::Patcher                                   patcher,
                            const galp::format::Expander                                  expander,
                            const uint32_t                                         n_samples);
uint64_t* decompress_column(const galp::codec::device::CROSSRLELaneMaskColumn<uint64_t> column,
                            const unsigned                                         unpack_n_vectors,
                            const unsigned                                         unpack_n_values,
                            const galp::format::Unpacker                                  unpacker,
                            const galp::format::Patcher                                   patcher,
                            const galp::format::Expander                                  expander,
                            const uint32_t                                         n_samples);
uint32_t* decompress_column(const galp::codec::device::CROSSRLEColumn<uint32_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const galp::format::Unpacker                          unpacker,
                            const galp::format::Patcher                           patcher,
                            const galp::format::Expander                          expander,
                            const uint32_t                                 n_samples);
uint64_t* decompress_column(const galp::codec::device::CROSSRLEColumn<uint64_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const galp::format::Unpacker                          unpacker,
                            const galp::format::Patcher                           patcher,
                            const galp::format::Expander                          expander,
                            const uint32_t                                 n_samples);
uint32_t* decompress_column(const galp::codec::device::FREQColumn<uint32_t> column,
                            const unsigned                             unpack_n_vectors,
                            const unsigned                             unpack_n_values,
                            const galp::format::Unpacker                      unpacker,
                            const galp::format::Patcher                       patcher,
                            const galp::format::Expander                      expander,
                            const uint32_t                             n_samples);
uint64_t* decompress_column(const galp::codec::device::FREQColumn<uint64_t> column,
                            const unsigned                             unpack_n_vectors,
                            const unsigned                             unpack_n_values,
                            const galp::format::Unpacker                      unpacker,
                            const galp::format::Patcher                       patcher,
                            const galp::format::Expander                      expander,
                            const uint32_t                             n_samples);
int8_t*   decompress_column(const galp::codec::device::FREQColumn<int8_t> column,
                            const unsigned                           unpack_n_vectors,
                            const unsigned                           unpack_n_values,
                            const galp::format::Unpacker                    unpacker,
                            const galp::format::Patcher                     patcher,
                            const galp::format::Expander                    expander,
                            const uint32_t                           n_samples);
int16_t*  decompress_column(const galp::codec::device::FREQColumn<int16_t> column,
                            const unsigned                            unpack_n_vectors,
                            const unsigned                            unpack_n_values,
                            const galp::format::Unpacker                     unpacker,
                            const galp::format::Patcher                      patcher,
                            const galp::format::Expander                     expander,
                            const uint32_t                            n_samples);
uint32_t* decompress_column(const galp::codec::device::FREQExtendedColumn<uint32_t> column,
                            const unsigned                                     unpack_n_vectors,
                            const unsigned                                     unpack_n_values,
                            const galp::format::Unpacker                              unpacker,
                            const galp::format::Patcher                               patcher,
                            const galp::format::Expander                              expander,
                            const uint32_t                                     n_samples);
uint64_t* decompress_column(const galp::codec::device::FREQExtendedColumn<uint64_t> column,
                            const unsigned                                     unpack_n_vectors,
                            const unsigned                                     unpack_n_values,
                            const galp::format::Unpacker                              unpacker,
                            const galp::format::Patcher                               patcher,
                            const galp::format::Expander                              expander,
                            const uint32_t                                     n_samples);
int8_t*   decompress_column(const galp::codec::device::FREQExtendedColumn<int8_t> column,
                            const unsigned                                   unpack_n_vectors,
                            const unsigned                                   unpack_n_values,
                            const galp::format::Unpacker                            unpacker,
                            const galp::format::Patcher                             patcher,
                            const galp::format::Expander                            expander,
                            const uint32_t                                   n_samples);
int16_t*  decompress_column(const galp::codec::device::FREQExtendedColumn<int16_t> column,
                            const unsigned                                    unpack_n_vectors,
                            const unsigned                                    unpack_n_values,
                            const galp::format::Unpacker                             unpacker,
                            const galp::format::Patcher                              patcher,
                            const galp::format::Expander                             expander,
                            const uint32_t                                    n_samples);
float*    decompress_column(const galp::codec::device::ALPColumn<float> column,
                            const unsigned                         unpack_n_vectors,
                            const unsigned                         unpack_n_values,
                            const galp::format::Unpacker                  unpacker,
                            const galp::format::Patcher                   patcher,
                            const galp::format::Expander                  expander,
                            const uint32_t                         n_samples);
double*   decompress_column(const galp::codec::device::ALPColumn<double> column,
                            const unsigned                          unpack_n_vectors,
                            const unsigned                          unpack_n_values,
                            const galp::format::Unpacker                   unpacker,
                            const galp::format::Patcher                    patcher,
                            const galp::format::Expander                   expander,
                            const uint32_t                          n_samples);
float*    decompress_column(const galp::codec::device::ALPExtendedColumn<float> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const galp::format::Unpacker                          unpacker,
                            const galp::format::Patcher                           patcher,
                            const galp::format::Expander                          expander,
                            const uint32_t                                 n_samples);
double*   decompress_column(const galp::codec::device::ALPExtendedColumn<double> column,
                            const unsigned                                  unpack_n_vectors,
                            const unsigned                                  unpack_n_values,
                            const galp::format::Unpacker                           unpacker,
                            const galp::format::Patcher                            patcher,
                            const galp::format::Expander                           expander,
                            const uint32_t                                  n_samples);

bool query_column(const galp::codec::device::BPColumn<uint32_t> column,
                  const unsigned                           unpack_n_vectors,
                  const unsigned                           unpack_n_values,
                  const galp::format::Unpacker                    unpacker,
                  const galp::format::Patcher                     patcher,
                  const uint32_t                           magic_value,
                  const uint32_t                           n_samples);
bool query_column(const galp::codec::device::BPColumn<uint64_t> column,
                  const unsigned                           unpack_n_vectors,
                  const unsigned                           unpack_n_values,
                  const galp::format::Unpacker                    unpacker,
                  const galp::format::Patcher                     patcher,
                  const uint64_t                           magic_value,
                  const uint32_t                           n_samples);
bool query_column(const galp::codec::device::FFORColumn<uint32_t> column,
                  const unsigned                             unpack_n_vectors,
                  const unsigned                             unpack_n_values,
                  const galp::format::Unpacker                      unpacker,
                  const galp::format::Patcher                       patcher,
                  const uint32_t                             magic_value,
                  const uint32_t                             n_samples);
bool query_column(const galp::codec::device::FFORColumn<uint64_t> column,
                  const unsigned                             unpack_n_vectors,
                  const unsigned                             unpack_n_values,
                  const galp::format::Unpacker                      unpacker,
                  const galp::format::Patcher                       patcher,
                  const uint64_t                             magic_value,
                  const uint32_t                             n_samples);
bool query_column(const galp::codec::device::ALPColumn<float> column,
                  const unsigned                         unpack_n_vectors,
                  const unsigned                         unpack_n_values,
                  const galp::format::Unpacker                  unpacker,
                  const galp::format::Patcher                   patcher,
                  const float                            magic_value,
                  const uint32_t                         n_samples);
bool query_column(const galp::codec::device::ALPColumn<double> column,
                  const unsigned                          unpack_n_vectors,
                  const unsigned                          unpack_n_values,
                  const galp::format::Unpacker                   unpacker,
                  const galp::format::Patcher                    patcher,
                  const double                            magic_value,
                  const uint32_t                          n_samples);
bool query_column(const galp::codec::device::ALPExtendedColumn<float> column,
                  const unsigned                                 unpack_n_vectors,
                  const unsigned                                 unpack_n_values,
                  const galp::format::Unpacker                          unpacker,
                  const galp::format::Patcher                           patcher,
                  const float                                    magic_value,
                  const uint32_t                                 n_samples);
bool query_column(const galp::codec::device::ALPExtendedColumn<double> column,
                  const unsigned                                  unpack_n_vectors,
                  const unsigned                                  unpack_n_values,
                  const galp::format::Unpacker                           unpacker,
                  const galp::format::Patcher                            patcher,
                  const double                                    magic_value,
                  const uint32_t                                  n_samples);

bool compute_column(const galp::codec::device::FFORColumn<uint32_t> column,
                    const unsigned                             unpack_n_vectors,
                    const unsigned                             unpack_n_values,
                    const galp::format::Unpacker                      unpacker,
                    const galp::format::Patcher                       patcher,
                    const unsigned                             n_repetitions,
                    const uint32_t                             n_samples);
bool compute_column(const galp::codec::device::FFORColumn<uint64_t> column,
                    const unsigned                             unpack_n_vectors,
                    const unsigned                             unpack_n_values,
                    const galp::format::Unpacker                      unpacker,
                    const galp::format::Patcher                       patcher,
                    const unsigned                             n_repetitions,
                    const uint32_t                             n_samples);

bool query_multi_column(const galp::codec::host::FFORColumn<uint32_t>& column,
                        const unsigned                           unpack_n_vectors,
                        const unsigned                           unpack_n_values,
                        const galp::format::Unpacker                    unpacker,
                        const galp::format::Patcher                     patcher,
                        const uint32_t                           magic_value,
                        const uint32_t                           n_samples);
bool query_multi_column(const galp::codec::host::FFORColumn<uint64_t>& column,
                        const unsigned                           unpack_n_vectors,
                        const unsigned                           unpack_n_values,
                        const galp::format::Unpacker                    unpacker,
                        const galp::format::Patcher                     patcher,
                        const uint64_t                           magic_value,
                        const uint32_t                           n_samples);
bool query_multi_column(const galp::codec::host::ALPColumn<float>& column,
                        const unsigned                       unpack_n_vectors,
                        const unsigned                       unpack_n_values,
                        const galp::format::Unpacker                unpacker,
                        const galp::format::Patcher                 patcher,
                        const float                          magic_value,
                        const uint32_t                       n_samples);
bool query_multi_column(const galp::codec::host::ALPColumn<double>& column,
                        const unsigned                        unpack_n_vectors,
                        const unsigned                        unpack_n_values,
                        const galp::format::Unpacker                 unpacker,
                        const galp::format::Patcher                  patcher,
                        const double                          magic_value,
                        const uint32_t                        n_samples);
bool query_multi_column(const galp::codec::host::ALPExtendedColumn<float>& column,
                        const unsigned                               unpack_n_vectors,
                        const unsigned                               unpack_n_values,
                        const galp::format::Unpacker                        unpacker,
                        const galp::format::Patcher                         patcher,
                        const float                                  magic_value,
                        const uint32_t                               n_samples);
bool query_multi_column(const galp::codec::host::ALPExtendedColumn<double>& column,
                        const unsigned                                unpack_n_vectors,
                        const unsigned                                unpack_n_values,
                        const galp::format::Unpacker                         unpacker,
                        const galp::format::Patcher                          patcher,
                        const double                                  magic_value,
                        const uint32_t                                n_samples);

} // namespace galp::bench::bindings

#endif // GENERATED_KERNEL_BINDINGS_CUH
