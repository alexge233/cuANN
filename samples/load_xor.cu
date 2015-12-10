#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <fstream>
#include <boost/archive/text_iarchive.hpp>

int main ()
{
    cuANN::ann xor_net;

    // create and open an archive for input
    std::ifstream ifs("xor_tanh_scaled.net");
    boost::archive::text_iarchive ia(ifs);

    // read class state from archive
    ia >> xor_net;

    // print on screen the loaded weights
    xor_net.print_weights();

    return 0;
}
