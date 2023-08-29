
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*                                                                           */
/*                                                                           */
/*                   MV++ Numerical Matrix/Vector C++ Library                */
/*                             MV++ Version 1.5                              */
/*                                                                           */
/*                                  R. Pozo                                  */
/*               National Institute of Standards and Technology              */
/*                                                                           */
/*                                  NOTICE                                   */
/*                                                                           */
/* Permission to use, copy, modify, and distribute this software and         */
/* its documentation for any purpose and without fee is hereby granted       */
/* provided that this permission notice appear in all copies and             */
/* supporting documentation.                                                 */
/*                                                                           */
/* Neither the Institution (National Institute of Standards and Technology)  */
/* nor the author makes any representations about the suitability of this    */
/* software for any purpose.  This software is provided ``as is''without     */
/* expressed or implied warranty.                                            */
/*                                                                           */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

//
//      mvmtp.h  : basic templated numerical matrix class, storage
//                  by columns (Fortran oriented.)
//
//
//

#ifndef _MV_MATRIX_TYPE_H_
#define _MV_MATRIX_TYPE_H_

#include "mv_vector_TYPE.h"
#include "mvmrf.h"

// for formatted printing of matrices
#include <sstream>
#ifdef MV_MATRIX_BOUNDS_CHECK
#include <assert.h>
#endif

class MV_ColMat_TYPE {
 private:
  MV_Vector_TYPE v_;
  int dim0_;  // perferred to using dim_[2]. some compilers
  int dim1_;  // refuse to initalize these in the constructor.
  int lda_;
  int ref_;  // true if this is declared as a reference vector,
             // i.e. it does not own the memory space, but
             // rather it is a view to another vector or array.
 public:
  /*::::::::::::::::::::::::::*/
  /* Constructors/Destructors */
  /*::::::::::::::::::::::::::*/

  MV_ColMat_TYPE();
  MV_ColMat_TYPE(int, int);

  // some compilers have difficulty with inlined 'for' statements.
  MV_ColMat_TYPE(int, int, const TYPE &);

  // usual copy by value
  // (can't use default parameter lda=m, because m is not a constant...)
  //
  MV_ColMat_TYPE(TYPE *, int m, int n);
  MV_ColMat_TYPE(TYPE *, int m, int n, int lda);

  // the "reference" versions
  //
  //
  MV_ColMat_TYPE(TYPE *, int m, int n, MV_Matrix_::ref_type i);
  MV_ColMat_TYPE(TYPE *, int m, int n, int lda, MV_Matrix_::ref_type i);

  MV_ColMat_TYPE(const MV_ColMat_TYPE &);
  ~MV_ColMat_TYPE();

  /*::::::::::::::::::::::::::::::::*/
  /*  Indices and access operations */
  /*::::::::::::::::::::::::::::::::*/

  inline TYPE &operator()(int, int);

  inline const TYPE &operator()(int, int) const;

  MV_ColMat_TYPE operator()(const MV_VecIndex &I, const MV_VecIndex &J);

  const MV_ColMat_TYPE operator()(const MV_VecIndex &I, const MV_VecIndex &J) const;

  int size(int i) const;

  MV_ColMat_TYPE &newsize(int, int);

  int ref() const { return ref_; }

  /*::::::::::::::*/
  /*  Assignment  */
  /*::::::::::::::*/

  MV_ColMat_TYPE &operator=(const MV_ColMat_TYPE &);

  MV_ColMat_TYPE &operator=(const TYPE &);

  friend std::ostream &operator<<(std::ostream &s, const MV_ColMat_TYPE &A);
};

inline TYPE &MV_ColMat_TYPE::operator()(int i, int j) {
#ifdef MV_MATRIX_BOUNDS_CHECK
  assert(0 <= i && i < size(0));
  assert(0 <= j && j < size(1));
#endif
  return v_(j * lda_ + i);  // could use indirect addressing
                            // instead...
}

inline const TYPE &MV_ColMat_TYPE::operator()(int i, int j) const {
#ifdef MV_MATRIX_BOUNDS_CHECK
  assert(0 <= i && i < size(0));
  assert(0 <= j && j < size(1));
#endif
  return v_(j * lda_ + i);
}

inline MV_ColMat_TYPE::MV_ColMat_TYPE(TYPE *d, int m, int n, MV_Matrix_::ref_type i)
    : v_(d, m * n, MV_Vector_::ref), dim0_(m), dim1_(n), lda_(m), ref_(i) {}

inline MV_ColMat_TYPE::MV_ColMat_TYPE(TYPE *d, int m, int n, int lda, MV_Matrix_::ref_type i)
    : v_(d, lda * n, MV_Vector_::ref), dim0_(m), dim1_(n), lda_(lda), ref_(i) {}

#endif
