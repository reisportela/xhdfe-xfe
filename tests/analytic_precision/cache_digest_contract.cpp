#include "cache_digest.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 3;
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    const int threads = std::stoi(argv[2]);
    auto print = [](const std::array<std::uint8_t,32>& value) {
        for (auto byte : value)
            std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(byte);
        std::cout << '\n';
    };
    hdfe::detail::CacheSha256 raw;
    raw.update(bytes.data(), bytes.size());
    print(raw.finish());
    hdfe::detail::CacheDigestBuilder tree(UINT64_C(0x5848444645544553));
    tree.word(17);
    tree.data(bytes.data(), bytes.size());
    tree.word(bytes.size());
    tree.data(bytes.empty() ? bytes.data() : bytes.data()+1, bytes.empty() ? 0 : bytes.size()-1);
    print(tree.finish(threads));
}
