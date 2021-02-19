#define BOOST_TEST_MODULE Conv2dTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../conv/conv_layer.hpp"

#define DEBUG true
#define DEBUG_PREFIX "[CONV LAYER TESTS ]\t"

BOOST_AUTO_TEST_CASE(ConstructorTest)
{
  Conv2d c(
      3,  // Input height.
      3,  // Input width.
      1,  // Input channels.
      2,  // kernel height.
      2,  // kernel width.
      2,  // output channels.
      1,  // Horizontal stride.
      1  // Vertical stride.
      );
}

BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  arma::cube input(3, 3, 1, arma::fill::zeros);
  input.slice(0) = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};

  arma::cube kernel1(2, 2, 1, arma::fill::zeros);
  kernel1.slice(0) = {{1, 0}, {0, 1}};

  arma::cube kernel2(2, 2, 1, arma::fill::zeros);
  kernel2.slice(0) = {{0, 1}, {1, 0}};

  std::vector<arma::cube> kernels;
  kernels.push_back(kernel1);
  kernels.push_back(kernel2);

  Conv2d c(
      3,  // Input height.
      3,  // Input width.
      1,  // Input channels.
      2,  // kernel height.
      2,  // kernel width.
      2,  // output channels.
      1,  // Horizontal stride.
      1  // Vertical stride.
      );

  c.set_kernels(kernels);

  arma::cube output = c.forward(input);
}

BOOST_AUTO_TEST_CASE(BackwardPassTest)
{
  arma::cube input(3, 3, 1, arma::fill::zeros);
  input.slice(0) = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};

  arma::cube kernel1(2, 2, 1, arma::fill::zeros);
  kernel1.slice(0) = {{1, 0}, {0, 1}};

  arma::cube kernel2(2, 2, 1, arma::fill::zeros);
  kernel2.slice(0) = {{0, 1}, {1, 0}};

  std::vector<arma::cube> kernels;
  kernels.push_back(kernel1);
  kernels.push_back(kernel2);

  Conv2d c(
      3,  // Input height.
      3,  // Input width.
      1,  // Input channels.
      2,  // kernel height.
      2,  // kernel width.
      2,  // output channels.
      1,  // Horizontal stride.
      1  // Vertical stride.
      );

  c.set_kernels(kernels);

  arma::cube output = c.forward(input);

  // For now, let the loss be the sum of all the output activations. Therefore,
  // the upstream gradient is all ones.
  arma::cube upstreamGradient(2, 2, 2, arma::fill::ones);

  c.backward(upstreamGradient, output);

  arma::cube gradInput = c.get_grad_wrt_input();

  std::vector<arma::cube> gradkernels = c.get_grad_wrt_kernels();

  // Now compute approximate gradients.
  double disturbance = 0.5e-5;

  output = arma::zeros(arma::size(output));
  arma::cube approxGradientWrtInput(arma::size(input), arma::fill::zeros);
  for (size_t i=0; i<input.n_elem; i++)
  {
    input[i] += disturbance;
    output = c.forward(input);
    double l1 = arma::accu(output);
    input[i] -= 2*disturbance;
    output = c.forward(input);
    double l2 = arma::accu(output);
    approxGradientWrtInput[i] = (l1 - l2)/(2.0*disturbance);
    input[i] += disturbance;
  }

#if DEBUG
  std::cout
      << DEBUG_PREFIX << "---------------------------------------------"
      << std::endl
      << DEBUG_PREFIX << "BACKWARD PASS TEST (BackwardPassTest) DEBUG OUTPUT"
      << std::endl
      << DEBUG_PREFIX << "---------------------------------------------"
      << std::endl;
  std::cout << DEBUG_PREFIX << "Approx gradient wrt inputs:" << std::endl;
  for (size_t s=0; s<approxGradientWrtInput.n_slices; s++)
  {
    std::cout << DEBUG_PREFIX << "Slice #" << s << std::endl;
    for (size_t r=0; r<approxGradientWrtInput.slice(s).n_rows; r++)
      std::cout << DEBUG_PREFIX << approxGradientWrtInput.slice(s).row(r);
  }
#endif
}

BOOST_AUTO_TEST_CASE(BackwardPassBigTest)
{
  // Input is 7 rows, 11 cols, and 3 slices.
  arma::cube input(7, 11, 3, arma::fill::randn);

  Conv2d c(
      7,  // Input height.
      11,  // Input width.
      3,  // Input channels.
      3,  // kernel height.
      5,  // kernel width.
      2,  // output channels.
      2,  // Horizontal stride.
      2  // Vertical stride.
      );
  arma::cube output = c.forward(input);

  // For now, let the loss be the sum of all the output activations. Therefore,
  // the upstream gradient is all ones.
  arma::cube upstreamGradient(3, 4, 2, arma::fill::ones);

  c.backward(upstreamGradient, output);

  arma::cube gradInput = c.get_grad_wrt_input();

  std::vector<arma::cube> gradkernels = c.get_grad_wrt_kernels();

  // Now compute approximate gradients.
  double disturbance = 0.5e-5;

  output = arma::zeros(arma::size(output));
  arma::cube approxGradientWrtInput(arma::size(input), arma::fill::zeros);
  for (size_t i=0; i<input.n_elem; i++)
  {
    input[i] += disturbance;
    output = c.forward(input);
    double l1 = arma::accu(output);
    input[i] -= 2*disturbance;
    output = c.forward(input);
    double l2 = arma::accu(output);
    approxGradientWrtInput[i] = (l1 - l2)/(2.0*disturbance);
    input[i] += disturbance;
  }

#if DEBUG
  std::cout
      << DEBUG_PREFIX << "---------------------------------------------"
      << std::endl
      << DEBUG_PREFIX << "BACKWARD PASS TEST (BackwardPassBigTest) DEBUG OUTPUT"
      << std::endl
      << DEBUG_PREFIX << "---------------------------------------------"
      << std::endl;
  std::cout << DEBUG_PREFIX << "Approx gradient wrt inputs:" << std::endl;
  for (size_t s=0; s<approxGradientWrtInput.n_slices; s++)
  {
    std::cout << DEBUG_PREFIX << "Slice #" << s << std::endl;
    for (size_t r=0; r<approxGradientWrtInput.slice(s).n_rows; r++)
      std::cout << DEBUG_PREFIX << approxGradientWrtInput.slice(s).row(r);
  }
#endif

  BOOST_REQUIRE(arma::approx_equal(gradInput,
                                   approxGradientWrtInput,
                                   "absdiff",
                                   disturbance));

  std::vector<arma::cube> approxGradientWrtkernels(2);
  approxGradientWrtkernels[0] = arma::zeros(3, 5, 3);
  approxGradientWrtkernels[1] = arma::zeros(3, 5, 3);

  std::vector<arma::cube> kernels = c.get_kernels();

  for (size_t fidx=0; fidx<2; fidx++)
  {
    for (size_t idx=0; idx<kernels[fidx].n_elem; idx++)
    {
      kernels[fidx][idx] += disturbance;
      c.set_kernels(kernels);
      output = c.forward(input);
      double l1 = arma::accu(output);
      kernels[fidx][idx] -= 2.0*disturbance;
      c.set_kernels(kernels);
      output = c.forward(input);
      double l2 = arma::accu(output);
      approxGradientWrtkernels[fidx][idx] = (l1-l2)/(2.0*disturbance);
      kernels[fidx][idx] += disturbance;
      c.set_kernels(kernels);
    }
  }

  for (size_t fidx=0; fidx<2; fidx++)
    BOOST_REQUIRE(arma::approx_equal(gradkernels[fidx],
                  approxGradientWrtkernels[fidx],
                  "absdiff",
                  disturbance));
}

#undef DEBUG
#undef DEBUG_PREFIX
