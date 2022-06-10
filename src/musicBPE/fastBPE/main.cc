#include "fastBPE.hpp"

using namespace std;
using namespace fastBPE;

void printUsage() {
  cerr
      << "usage: music_bpe_exec <command> <args>\n\n"
      << "The commands supported by fastBPE is:\n\n"
      << "learnbpe nCodes input1 [input2]      learn BPE codes from one or two "
         "text files\n"
      << endl;
}


int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  string command = argv[1];
  if (command == "learnbpe") {
    assert(argc == 4 || argc == 5);
    learnbpe(stoi(argv[2]), argv[3], argc == 5 ? argv[4] : "");
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}
