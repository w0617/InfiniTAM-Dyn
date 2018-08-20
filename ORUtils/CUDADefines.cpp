#include "CUDADefines.h"

#ifdef WITH_BACKWARDS_CPP
#include "third_party/backward-cpp/backward.hpp"
#endif

namespace ORUtils {

void __cudaSafeCall(cudaError err, const char *file, const int line) {
	if (cudaSuccess != err) {

#ifdef WITH_BACKWARDS_CPP
        using namespace backward;
        const int kStackTraceDepth = 32;
        // The number of top entries to skip when printing the stack trace.
        // These include two internal layers from `backward-cpp`, plus this current function.
        const int kStackTraceSkip = 0;

        fprintf(stderr, "\nCUDA error. See stacktrace and details below:\n\n");

        // Display a helpful backtrace (with code snippets, if available).
        StackTrace st;
        st.load_here(kStackTraceDepth);

        // Disable printing out boilerplate stack frames from the stack trace
        // processing code.
        st.skip_n_firsts(kStackTraceSkip);

        // TODO(andrei): Make the printer use $source($line) ... format, so that the err positions
        // become clickable in CLion.
        Printer p;
        p.address = true;

        // As of July 2017, this sometimes crashes while printing the stack trace, because of course
        // it does. So some CUDA errors may be reported as a simple segfault, since the printer
        // segfaults while attempting to print the true stack trace.
        p.print(st);
        fprintf(stderr, "\n");
#endif

		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %d | %s.\n",
				file, line, err, cudaGetErrorString(err));
		fflush(stderr);

		exit(-1);
	}
}

}
