#pragma once

#include <algorithm>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h> // ftruncate
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace fastBPE {

using namespace std;

const size_t kMaxPairs = 1000 * 1000 * 1000;


int safeOpen(const char *file_path, int flags, mode_t mode = 0) {
  int fd = open(file_path, flags, mode);
  if (fd < 0) {
    fprintf(stderr, "Cannot open text file %s\n", file_path);
    exit(EXIT_FAILURE);
  }
  return fd;
}
char pit2chr(int pit) {
    return char(pit);
}
void readText(const char *fp, unordered_map<string, uint32_t> &word_count) {
  char cur_word[300];

  uint64_t total = 0;

    FILE* fin = fopen(fp, "r");
    int tot;
    fscanf(fin, "%d", &tot);
    fprintf(stderr, "total %d words...\n", tot);
    for (int i = 0; i < tot; i++) {
        fscanf(fin, "%s", cur_word);
        int cnt;
        fscanf(fin, "%d", &cnt);
        string s(cur_word);
        word_count[s] = cnt;
        total += cnt;
    }
   

  fprintf(stderr, "Read %lu words (%lu unique) from text file.\n", total,
          word_count.size());
}

std::pair<size_t, uint64_t> output_or_count(
  unordered_map<string, string> &bpe, size_t size, char *f, char *fo
) {
  string cur_word;
  size_t charOut = 0;
  uint64_t total = 0;
  for (size_t i = 0; i < size; i++) {
    auto &cur_char = f[i];
    if (cur_char == ' ' || cur_char == '\n') {
      if (cur_word.size() == 0) {
        if (fo != nullptr) fo[charOut] = cur_char;
        charOut++;
        continue;
      }
      // end of word : write bpe to output
      auto it = bpe.find(cur_word);
      assert(it != bpe.end());
      for (auto x : it->second) {
        if (fo != nullptr) fo[charOut] = x;
        charOut++;
      }
      if (fo != nullptr) fo[charOut] = cur_char;
      charOut++;

      total++;
      cur_word.clear();
    } else {
      cur_word.push_back(cur_char);
    }
  }
  return std::make_pair(charOut, total);
}

void outputText(const char *fpo, const char *fp,
                unordered_map<string, string> &bpe) {

  int fd = safeOpen(fp, O_RDONLY);
  auto fdOut = safeOpen(fpo, O_RDWR | O_CREAT | O_TRUNC, 0666);

  struct stat s;
  fstat(fd, &s);

  fprintf(stderr, "Applying BPE to %s ...\n", fp);
  auto size = s.st_size;
  char *f = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

  auto p = output_or_count(bpe, size, f, nullptr);
  size_t out_size = p.first;

  if (ftruncate(fdOut, out_size) < 0) {
    fprintf(stderr, "Couldn't truncate output file %s to size %lu\n", fpo,
            out_size);
    exit(EXIT_FAILURE);
  }


  char *fo = (char *)mmap(NULL, out_size, PROT_WRITE, MAP_SHARED, fdOut, 0);
  if (fo == MAP_FAILED) {
    fprintf(stderr, "Output memory map failed : %d.\n", errno);
    exit(EXIT_FAILURE);
  }
  p = output_or_count(bpe, size, f, fo);
  fprintf(stderr, "Modified %lu words from text file.\n", p.second);
  munmap(fo, out_size);
  munmap(f, size);
  close(fdOut);
  close(fd);
}

struct pair_hash {
  template <class T1, class T2> size_t operator()(const pair<T1, T2> &p) const {
    auto h1 = hash<T1>{}(p.first);
    auto h2 = hash<T2>{}(p.second);
    size_t seed = h1;
    // boost::hash_combine
    return h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};
char pit2alphabet[] = {'C', 'd', 'D', 'e', 'E', 'F', 'g', 'G', 'a', 'A', 'b', 'B'};
char oct2alphabet[] = {'O', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
void build_init_vocabs(unordered_map<string, uint32_t> &token_to_int, 
                       vector<string> &int_to_token) {

    for(int i = 0; i < 128; i++) {
        int octave = i / 12;
        int relpit = i % 12;
        string oct(1, oct2alphabet[octave]);
        string rel(1, pit2alphabet[relpit]);
        string s = rel + oct;
        int_to_token.push_back(s);
        token_to_int[s] = i;
    }
}
void tokenize(const unordered_map<string, uint32_t> &word_count,
              unordered_map<string, uint32_t> &token_to_int,
              vector<string> &int_to_token, vector<list<uint32_t>> &words,
              vector<int32_t> &counts) {
  build_init_vocabs(token_to_int, int_to_token);
  
  for (auto &x : word_count) {
    auto &word = x.first;

    words.push_back(list<uint32_t>());
    auto &current_word = words.back();
    counts.push_back(x.second);

    int tmplen = word.length();
    uint32_t lasttoken = -1;
    for (int i = 0; i < tmplen; i += 2) {
        auto new_token = word.substr(i, 2);
        if (i + 2 == tmplen)
            new_token = new_token;// + kEndWord;
        
        if (token_to_int.count(new_token) == 0) {
            fprintf(stderr, "Error init vocab: %s\n", (char*)new_token.c_str());
            exit(0);
        }
        uint32_t cur_pit = token_to_int[new_token];
        if (lasttoken != -1 && cur_pit <= lasttoken) {
            fprintf(stderr, "Error init vocab order: in %s\n", (char*)word.c_str());
            exit(0);
        }
        lasttoken = cur_pit;
        current_word.push_back(cur_pit);
    }
  }
}

using tp = pair<uint32_t, uint32_t>;
using tps = pair<string, string>;
using pc = unordered_map<tp, pair<int32_t, tp> *, pair_hash>;
// process every word
void count_in_word(
    list<uint32_t> &word, uint32_t wi, uint32_t count, pc &pair_counts,
    vector<pair<int32_t, tp>> &contiguous_counts,
    unordered_map<tp, unordered_set<uint32_t>, pair_hash> &where) {

  tp cur_pair;

  for (auto i = word.begin(); i != word.end(); ++i) {
      auto tmp = i;
      ++tmp;
      for (auto j = tmp; j != word.end(); ++j) {
          cur_pair.first = *i;
          cur_pair.second = *j;
          auto it = pair_counts.find(cur_pair);
          if (it == pair_counts.end()) {
              contiguous_counts.emplace_back(0, cur_pair);
              auto *added = &contiguous_counts.back();
              pair_counts.emplace(piecewise_construct, forward_as_tuple(cur_pair), forward_as_tuple(added));
              where[cur_pair] = unordered_set<uint32_t>();
          }
          if (count > 0) {where[cur_pair].insert(wi);} else {fprintf(stderr, "count in word init error!\n"); exit(0);}
          pair_counts[cur_pair]->first += count;
      }
  }
}

void find_maxp(vector<pair<int32_t, tp>> &contiguous_counts, tp &maxp,
               int32_t &max_c) {
  max_c = 0;
  for (auto &x : contiguous_counts) {
    if (x.first > max_c) {
      max_c = x.first;
      maxp = x.second;
    } else if (x.first == max_c and x.second < maxp) {
      maxp = x.second;
    }
  }
}


void learnbpe(const uint32_t kNPairs, const char *inputFile1,
              const char *inputFile2) {
  // get vocab
  unordered_map<string, uint32_t> word_count;
  readText(inputFile1, word_count);
  if (strcmp(inputFile2, "") != 0) {
    readText(inputFile2, word_count);
  }

  // a token is an int, it represents a string
  unordered_map<string, uint32_t> token_to_int;
  vector<string> int_to_token;

  vector<list<uint32_t>> words;
  vector<int32_t> counts;

  tokenize(word_count, token_to_int, int_to_token, words, counts);
    
 // tp: pair<int, int>
  vector<pair<int32_t, tp>> contiguous_counts;
  contiguous_counts.reserve(kMaxPairs);

  pc pair_counts;
  unordered_map<tp, unordered_set<uint32_t>, pair_hash> where_to_update;

  tp cur_pair;
  int32_t max_c = 0;
  tp max_p;
  for (uint32_t wi = 0; wi < words.size(); wi++) {
    count_in_word(words[wi], wi, counts[wi], pair_counts, contiguous_counts,
                  where_to_update);
  }

  
  for (size_t i = 0; i < kNPairs; i++) {
    // create new token for pair. replace
    find_maxp(contiguous_counts, max_p, max_c);

    auto new_token = int_to_token[max_p.first] + int_to_token[max_p.second];
    cout << int_to_token[max_p.first] << " " << int_to_token[max_p.second]
        << " " << max_c << endl;

    uint32_t new_token_id = int_to_token.size();
    int_to_token.push_back(new_token);
    token_to_int[new_token] = new_token_id;
    max_c = 0;
    auto change_count = [&](tp pair, int32_t v, uint32_t wi) {
      auto it = pair_counts.find(pair);
      if (it != pair_counts.end()) {
        it->second->first += v;
      } else {
        if (v > 0) {
          contiguous_counts.emplace_back(v, pair);
          pair_counts.emplace(piecewise_construct, forward_as_tuple(pair),
                              forward_as_tuple(&(contiguous_counts.back())));
          where_to_update[pair] = unordered_set<uint32_t>();
        }
      }
      if (v > 0)
        where_to_update[pair].insert(wi);
      else
        where_to_update[pair].erase(wi);
    };

    for (auto wi : where_to_update[max_p]) {
        
      auto &cur_word = words[wi];
      int cnt = 0;
      for (auto it = cur_word.begin(); it != cur_word.end(); ++it) {
          if (*it == max_p.first || *it == max_p.second) 
              cnt++;
      }
      if (cnt != 2) {  // where to update is not maintained
          fprintf(stderr, "where to update is not maintained! %d %d %d    ", cnt, max_p.first, max_p.second);
          for (auto it = cur_word.begin(); it != cur_word.end(); ++it)
              fprintf(stderr, "%d ", *it);
          fprintf(stderr, "\n");
          continue;
      }
          
    
      auto it = cur_word.begin();
      while (it != cur_word.end()) {
        if (*it != max_p.first && *it != max_p.second) {
            uint32_t u, v;
            
            u = *it;
            v = max_p.first;
            if (u > v) swap(u, v);
            change_count(make_pair(u, v), -counts[wi], wi);
            
            u = *it;
            v = max_p.second;
            if (u > v) swap(u, v);
            change_count(make_pair(u, v), -counts[wi], wi);
            
            u = *it;
            v = new_token_id;
            if (u > v) swap(u, v);
            change_count(make_pair(u, v), counts[wi], wi);
            ++it;
        } else 
            it = cur_word.erase(it);
        
      }
      cur_word.insert(cur_word.end(), new_token_id);
    }
    

    if (pair_counts.find(max_p) != pair_counts.end()){
      pair_counts[max_p]->first = 0;
    }
    
  }
  for (int i = 0; i < int_to_token.size(); ++i) {
    fprintf(stderr, "%d %s\n", i, int_to_token[i].c_str());
  }
}

void split(vector<string> &splits, const string &text, char sep) {
  size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != string::npos) {
    if (end != start)
      splits.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  if (end != start && start < text.size())
    splits.push_back(text.substr(start));
}

void readVocab(const char *fp, unordered_map<string, uint32_t> &vocab) {
  ifstream file(fp);
  if (!file) {
    fprintf(stderr, "Cannot open vocabulary file %s\n", fp);
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Loading vocabulary from %s ...\n", fp);
  string line;
  uint64_t total = 0;
  while (getline(file, line)) {
    vector<string> splits;
    split(splits, line, ' ');
    assert(splits.size() == 2);
    assert(vocab.find(splits[0]) == vocab.end());
    int count = stoi(splits[1]);
    vocab[splits[0]] = count;
    total += count;
  }
  fprintf(stderr, "Read %lu words (%lu unique) from vocabulary file.\n", total,
          vocab.size());
}


};