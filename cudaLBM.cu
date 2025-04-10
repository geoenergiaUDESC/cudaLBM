#include <iostream>
#include <stdint.h>

# ifdef SCALAR_PRECISION_32
typedef float scalar_t;
#elif SCALAR_PRECISION_64
typedef double scalar_t;
#endif

#ifdef LABEL_SIZE_32
typedef uint32_t label_t;
#elif LABEL_SIZE_64
typedef std::size_t label_t;
#endif

int main(void)
{
    std::cout << "Hello world!" << std::endl;

    std::cout << sizeof(label_t) << std::endl;

    return 0;
}