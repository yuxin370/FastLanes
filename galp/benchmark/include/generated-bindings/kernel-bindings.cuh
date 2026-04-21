// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/include/generated-bindings/kernel-bindings.cuh
// ────────────────────────────────────────────────────────
#ifndef GENERATED_KERNEL_BINDINGS_CUH
#define GENERATED_KERNEL_BINDINGS_CUH

#include "engine/enums.cuh"
#include "flsgpu/flsgpu.cuh"
#include <cstdint>

namespace bindings {

template <typename T, typename ColumnT>
T* decompress_column(const ColumnT         column,
                     const unsigned        unpack_n_vectors,
                     const unsigned        unpack_n_values,
                     const enums::Unpacker unpacker,
                     const enums::Patcher  patcher,
                     const enums::Expander expander,
                     const uint32_t        n_samples);

template <typename T, typename ColumnT>
T* decompress_column(const ColumnT         column,
                     const unsigned        unpack_n_vectors,
                     const unsigned        unpack_n_values,
                     const enums::Unpacker unpacker,
                     const enums::Patcher  patcher,
                     const enums::Expander expander,
                     const uint32_t        n_samples,
                     const bool            use_shuffle);

template <typename T, typename ColumnT>
bool query_column(const ColumnT         column,
                  const unsigned        unpack_n_vectors,
                  const unsigned        unpack_n_values,
                  const enums::Unpacker unpacker,
                  const enums::Patcher  patcher,
                  const T               magic_value,
                  const uint32_t        n_samples);

template <typename T, typename ColumnT>
bool compute_column(const ColumnT         column,
                    const unsigned        unpack_n_vectors,
                    const unsigned        unpack_n_values,
                    const enums::Unpacker unpacker,
                    const enums::Patcher  patcher,
                    const unsigned        n_repetitions,
                    const uint32_t        n_samples);

template <typename T, typename ColumnT>
bool query_multi_column(ColumnT               column,
                        const unsigned        unpack_n_vectors,
                        const unsigned        unpack_n_values,
                        const enums::Unpacker unpacker,
                        const enums::Patcher  patcher,
                        const T               magic_value,
                        const uint32_t        n_samples);

uint32_t* decompress_column(const flsgpu::device::BPColumn<uint32_t> column,
                            const unsigned                           unpack_n_vectors,
                            const unsigned                           unpack_n_values,
                            const enums::Unpacker                    unpacker,
                            const enums::Patcher                     patcher,
                            const enums::Expander                    expander,
                            const uint32_t                           n_samples);
uint64_t* decompress_column(const flsgpu::device::BPColumn<uint64_t> column,
                            const unsigned                           unpack_n_vectors,
                            const unsigned                           unpack_n_values,
                            const enums::Unpacker                    unpacker,
                            const enums::Patcher                     patcher,
                            const enums::Expander                    expander,
                            const uint32_t                           n_samples);
uint32_t* decompress_column(const flsgpu::device::FFORColumn<uint32_t> column,
                            const unsigned                             unpack_n_vectors,
                            const unsigned                             unpack_n_values,
                            const enums::Unpacker                      unpacker,
                            const enums::Patcher                       patcher,
                            const enums::Expander                      expander,
                            const uint32_t                             n_samples);
uint64_t* decompress_column(const flsgpu::device::FFORColumn<uint64_t> column,
                            const unsigned                             unpack_n_vectors,
                            const unsigned                             unpack_n_values,
                            const enums::Unpacker                      unpacker,
                            const enums::Patcher                       patcher,
                            const enums::Expander                      expander,
                            const uint32_t                             n_samples);
uint32_t* decompress_column(const flsgpu::device::SLPATCHColumn<uint32_t> column,
                            const unsigned                                unpack_n_vectors,
                            const unsigned                                unpack_n_values,
                            const enums::Unpacker                         unpacker,
                            const enums::Patcher                          patcher,
                            const enums::Expander                         expander,
                            const uint32_t                                n_samples);
uint64_t* decompress_column(const flsgpu::device::SLPATCHColumn<uint64_t> column,
                            const unsigned                                unpack_n_vectors,
                            const unsigned                                unpack_n_values,
                            const enums::Unpacker                         unpacker,
                            const enums::Patcher                          patcher,
                            const enums::Expander                         expander,
                            const uint32_t                                n_samples);
int16_t*  decompress_column(const flsgpu::device::SLPATCHColumn<int16_t> column,
                            const unsigned                               unpack_n_vectors,
                            const unsigned                               unpack_n_values,
                            const enums::Unpacker                        unpacker,
                            const enums::Patcher                         patcher,
                            const enums::Expander                        expander,
                            const uint32_t                               n_samples);
uint32_t* decompress_column(const flsgpu::device::DICTSLPATCHColumn<uint32_t> column,
                            const unsigned                                    unpack_n_vectors,
                            const unsigned                                    unpack_n_values,
                            const enums::Unpacker                             unpacker,
                            const enums::Patcher                              patcher,
                            const enums::Expander                             expander,
                            const uint32_t                                    n_samples);
uint64_t* decompress_column(const flsgpu::device::DICTSLPATCHColumn<uint64_t> column,
                            const unsigned                                    unpack_n_vectors,
                            const unsigned                                    unpack_n_values,
                            const enums::Unpacker                             unpacker,
                            const enums::Patcher                              patcher,
                            const enums::Expander                             expander,
                            const uint32_t                                    n_samples);
uint32_t* decompress_column(const flsgpu::device::RLEColumn<uint32_t, uint32_t> column,
                            const unsigned                                      unpack_n_vectors,
                            const unsigned                                      unpack_n_values,
                            const enums::Unpacker                               unpacker,
                            const enums::Patcher                                patcher,
                            const enums::Expander                               expander,
                            const uint32_t                                      n_samples);
uint64_t* decompress_column(const flsgpu::device::RLEColumn<uint64_t, uint64_t> column,
                            const unsigned                                      unpack_n_vectors,
                            const unsigned                                      unpack_n_values,
                            const enums::Unpacker                               unpacker,
                            const enums::Patcher                                patcher,
                            const enums::Expander                               expander,
                            const uint32_t                                      n_samples);
uint32_t* decompress_column(const flsgpu::device::CONSTANTColumn<uint32_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const enums::Unpacker                          unpacker,
                            const enums::Patcher                           patcher,
                            const enums::Expander                          expander,
                            const uint32_t                                 n_samples);
uint64_t* decompress_column(const flsgpu::device::CONSTANTColumn<uint64_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const enums::Unpacker                          unpacker,
                            const enums::Patcher                           patcher,
                            const enums::Expander                          expander,
                            const uint32_t                                 n_samples);
uint32_t* decompress_column(const flsgpu::device::DICTFFORColumn<uint32_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const enums::Unpacker                          unpacker,
                            const enums::Patcher                           patcher,
                            const enums::Expander                          expander,
                            const uint32_t                                 n_samples,
                            const bool                                     use_shuffle);
uint64_t* decompress_column(const flsgpu::device::DICTFFORColumn<uint64_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const enums::Unpacker                          unpacker,
                            const enums::Patcher                           patcher,
                            const enums::Expander                          expander,
                            const uint32_t                                 n_samples,
                            const bool                                     use_shuffle);
uint32_t* decompress_column(const flsgpu::device::CROSSRLEExtendedColumn<uint32_t> column,
                            const unsigned                                         unpack_n_vectors,
                            const unsigned                                         unpack_n_values,
                            const enums::Unpacker                                  unpacker,
                            const enums::Patcher                                   patcher,
                            const enums::Expander                                  expander,
                            const uint32_t                                         n_samples);
uint64_t* decompress_column(const flsgpu::device::CROSSRLEExtendedColumn<uint64_t> column,
                            const unsigned                                         unpack_n_vectors,
                            const unsigned                                         unpack_n_values,
                            const enums::Unpacker                                  unpacker,
                            const enums::Patcher                                   patcher,
                            const enums::Expander                                  expander,
                            const uint32_t                                         n_samples);
uint32_t* decompress_column(const flsgpu::device::CROSSRLELaneMaskColumn<uint32_t> column,
                            const unsigned                                         unpack_n_vectors,
                            const unsigned                                         unpack_n_values,
                            const enums::Unpacker                                  unpacker,
                            const enums::Patcher                                   patcher,
                            const enums::Expander                                  expander,
                            const uint32_t                                         n_samples);
uint64_t* decompress_column(const flsgpu::device::CROSSRLELaneMaskColumn<uint64_t> column,
                            const unsigned                                         unpack_n_vectors,
                            const unsigned                                         unpack_n_values,
                            const enums::Unpacker                                  unpacker,
                            const enums::Patcher                                   patcher,
                            const enums::Expander                                  expander,
                            const uint32_t                                         n_samples);
uint32_t* decompress_column(const flsgpu::device::CROSSRLEColumn<uint32_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const enums::Unpacker                          unpacker,
                            const enums::Patcher                           patcher,
                            const enums::Expander                          expander,
                            const uint32_t                                 n_samples);
uint64_t* decompress_column(const flsgpu::device::CROSSRLEColumn<uint64_t> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const enums::Unpacker                          unpacker,
                            const enums::Patcher                           patcher,
                            const enums::Expander                          expander,
                            const uint32_t                                 n_samples);
uint32_t* decompress_column(const flsgpu::device::FREQColumn<uint32_t> column,
                            const unsigned                             unpack_n_vectors,
                            const unsigned                             unpack_n_values,
                            const enums::Unpacker                      unpacker,
                            const enums::Patcher                       patcher,
                            const enums::Expander                      expander,
                            const uint32_t                             n_samples);
uint64_t* decompress_column(const flsgpu::device::FREQColumn<uint64_t> column,
                            const unsigned                             unpack_n_vectors,
                            const unsigned                             unpack_n_values,
                            const enums::Unpacker                      unpacker,
                            const enums::Patcher                       patcher,
                            const enums::Expander                      expander,
                            const uint32_t                             n_samples);
int8_t*   decompress_column(const flsgpu::device::FREQColumn<int8_t> column,
                            const unsigned                           unpack_n_vectors,
                            const unsigned                           unpack_n_values,
                            const enums::Unpacker                    unpacker,
                            const enums::Patcher                     patcher,
                            const enums::Expander                    expander,
                            const uint32_t                           n_samples);
int16_t*  decompress_column(const flsgpu::device::FREQColumn<int16_t> column,
                            const unsigned                            unpack_n_vectors,
                            const unsigned                            unpack_n_values,
                            const enums::Unpacker                     unpacker,
                            const enums::Patcher                      patcher,
                            const enums::Expander                     expander,
                            const uint32_t                            n_samples);
uint32_t* decompress_column(const flsgpu::device::FREQExtendedColumn<uint32_t> column,
                            const unsigned                                     unpack_n_vectors,
                            const unsigned                                     unpack_n_values,
                            const enums::Unpacker                              unpacker,
                            const enums::Patcher                               patcher,
                            const enums::Expander                              expander,
                            const uint32_t                                     n_samples);
uint64_t* decompress_column(const flsgpu::device::FREQExtendedColumn<uint64_t> column,
                            const unsigned                                     unpack_n_vectors,
                            const unsigned                                     unpack_n_values,
                            const enums::Unpacker                              unpacker,
                            const enums::Patcher                               patcher,
                            const enums::Expander                              expander,
                            const uint32_t                                     n_samples);
int8_t*   decompress_column(const flsgpu::device::FREQExtendedColumn<int8_t> column,
                            const unsigned                                   unpack_n_vectors,
                            const unsigned                                   unpack_n_values,
                            const enums::Unpacker                            unpacker,
                            const enums::Patcher                             patcher,
                            const enums::Expander                            expander,
                            const uint32_t                                   n_samples);
int16_t*  decompress_column(const flsgpu::device::FREQExtendedColumn<int16_t> column,
                            const unsigned                                    unpack_n_vectors,
                            const unsigned                                    unpack_n_values,
                            const enums::Unpacker                             unpacker,
                            const enums::Patcher                              patcher,
                            const enums::Expander                             expander,
                            const uint32_t                                    n_samples);
float*    decompress_column(const flsgpu::device::ALPColumn<float> column,
                            const unsigned                         unpack_n_vectors,
                            const unsigned                         unpack_n_values,
                            const enums::Unpacker                  unpacker,
                            const enums::Patcher                   patcher,
                            const enums::Expander                  expander,
                            const uint32_t                         n_samples);
double*   decompress_column(const flsgpu::device::ALPColumn<double> column,
                            const unsigned                          unpack_n_vectors,
                            const unsigned                          unpack_n_values,
                            const enums::Unpacker                   unpacker,
                            const enums::Patcher                    patcher,
                            const enums::Expander                   expander,
                            const uint32_t                          n_samples);
float*    decompress_column(const flsgpu::device::ALPExtendedColumn<float> column,
                            const unsigned                                 unpack_n_vectors,
                            const unsigned                                 unpack_n_values,
                            const enums::Unpacker                          unpacker,
                            const enums::Patcher                           patcher,
                            const enums::Expander                          expander,
                            const uint32_t                                 n_samples);
double*   decompress_column(const flsgpu::device::ALPExtendedColumn<double> column,
                            const unsigned                                  unpack_n_vectors,
                            const unsigned                                  unpack_n_values,
                            const enums::Unpacker                           unpacker,
                            const enums::Patcher                            patcher,
                            const enums::Expander                           expander,
                            const uint32_t                                  n_samples);

bool query_column(const flsgpu::device::BPColumn<uint32_t> column,
                  const unsigned                           unpack_n_vectors,
                  const unsigned                           unpack_n_values,
                  const enums::Unpacker                    unpacker,
                  const enums::Patcher                     patcher,
                  const uint32_t                           magic_value,
                  const uint32_t                           n_samples);
bool query_column(const flsgpu::device::BPColumn<uint64_t> column,
                  const unsigned                           unpack_n_vectors,
                  const unsigned                           unpack_n_values,
                  const enums::Unpacker                    unpacker,
                  const enums::Patcher                     patcher,
                  const uint64_t                           magic_value,
                  const uint32_t                           n_samples);
bool query_column(const flsgpu::device::FFORColumn<uint32_t> column,
                  const unsigned                             unpack_n_vectors,
                  const unsigned                             unpack_n_values,
                  const enums::Unpacker                      unpacker,
                  const enums::Patcher                       patcher,
                  const uint32_t                             magic_value,
                  const uint32_t                             n_samples);
bool query_column(const flsgpu::device::FFORColumn<uint64_t> column,
                  const unsigned                             unpack_n_vectors,
                  const unsigned                             unpack_n_values,
                  const enums::Unpacker                      unpacker,
                  const enums::Patcher                       patcher,
                  const uint64_t                             magic_value,
                  const uint32_t                             n_samples);
bool query_column(const flsgpu::device::ALPColumn<float> column,
                  const unsigned                         unpack_n_vectors,
                  const unsigned                         unpack_n_values,
                  const enums::Unpacker                  unpacker,
                  const enums::Patcher                   patcher,
                  const float                            magic_value,
                  const uint32_t                         n_samples);
bool query_column(const flsgpu::device::ALPColumn<double> column,
                  const unsigned                          unpack_n_vectors,
                  const unsigned                          unpack_n_values,
                  const enums::Unpacker                   unpacker,
                  const enums::Patcher                    patcher,
                  const double                            magic_value,
                  const uint32_t                          n_samples);
bool query_column(const flsgpu::device::ALPExtendedColumn<float> column,
                  const unsigned                                 unpack_n_vectors,
                  const unsigned                                 unpack_n_values,
                  const enums::Unpacker                          unpacker,
                  const enums::Patcher                           patcher,
                  const float                                    magic_value,
                  const uint32_t                                 n_samples);
bool query_column(const flsgpu::device::ALPExtendedColumn<double> column,
                  const unsigned                                  unpack_n_vectors,
                  const unsigned                                  unpack_n_values,
                  const enums::Unpacker                           unpacker,
                  const enums::Patcher                            patcher,
                  const double                                    magic_value,
                  const uint32_t                                  n_samples);

bool compute_column(const flsgpu::device::FFORColumn<uint32_t> column,
                    const unsigned                             unpack_n_vectors,
                    const unsigned                             unpack_n_values,
                    const enums::Unpacker                      unpacker,
                    const enums::Patcher                       patcher,
                    const unsigned                             n_repetitions,
                    const uint32_t                             n_samples);
bool compute_column(const flsgpu::device::FFORColumn<uint64_t> column,
                    const unsigned                             unpack_n_vectors,
                    const unsigned                             unpack_n_values,
                    const enums::Unpacker                      unpacker,
                    const enums::Patcher                       patcher,
                    const unsigned                             n_repetitions,
                    const uint32_t                             n_samples);

bool query_multi_column(const flsgpu::host::FFORColumn<uint32_t> column,
                        const unsigned                           unpack_n_vectors,
                        const unsigned                           unpack_n_values,
                        const enums::Unpacker                    unpacker,
                        const enums::Patcher                     patcher,
                        const uint32_t                           magic_value,
                        const uint32_t                           n_samples);
bool query_multi_column(const flsgpu::host::FFORColumn<uint64_t> column,
                        const unsigned                           unpack_n_vectors,
                        const unsigned                           unpack_n_values,
                        const enums::Unpacker                    unpacker,
                        const enums::Patcher                     patcher,
                        const uint64_t                           magic_value,
                        const uint32_t                           n_samples);
bool query_multi_column(const flsgpu::host::ALPColumn<float> column,
                        const unsigned                       unpack_n_vectors,
                        const unsigned                       unpack_n_values,
                        const enums::Unpacker                unpacker,
                        const enums::Patcher                 patcher,
                        const float                          magic_value,
                        const uint32_t                       n_samples);
bool query_multi_column(const flsgpu::host::ALPColumn<double> column,
                        const unsigned                        unpack_n_vectors,
                        const unsigned                        unpack_n_values,
                        const enums::Unpacker                 unpacker,
                        const enums::Patcher                  patcher,
                        const double                          magic_value,
                        const uint32_t                        n_samples);
bool query_multi_column(const flsgpu::host::ALPExtendedColumn<float> column,
                        const unsigned                               unpack_n_vectors,
                        const unsigned                               unpack_n_values,
                        const enums::Unpacker                        unpacker,
                        const enums::Patcher                         patcher,
                        const float                                  magic_value,
                        const uint32_t                               n_samples);
bool query_multi_column(const flsgpu::host::ALPExtendedColumn<double> column,
                        const unsigned                                unpack_n_vectors,
                        const unsigned                                unpack_n_values,
                        const enums::Unpacker                         unpacker,
                        const enums::Patcher                          patcher,
                        const double                                  magic_value,
                        const uint32_t                                n_samples);

} // namespace bindings

#endif // GENERATED_KERNEL_BINDINGS_CUH
