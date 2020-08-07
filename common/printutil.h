#ifndef __print_1337
#define __print_1337

#include <stdio.h>
#include <stdlib.h>

static const char RST[] = "\033[0m";
static const char BOLD[] = "\033[1m";
static const char R[] = "\033[31m";
static const char G[] = "\033[32m";
static const char Y[] = "\033[33m";


static void printInner(int type, const char * str) {
  if (type < 0){
    fprintf(stderr, "%s[x]%s %s, Exit..\n", R, RST, str);
    exit(1);
  } else if (type == 0) {
    fprintf(stdout, "%s[+]%s %s\n", G, RST, str);
  } else {
    fprintf(stdout, "%s[-]%s %s\n", Y, RST, str);
  }
}


static void printErr(const char * str) {
  printInner(-1, str);
}

static void printY(const char * str) {
  printInner(1, str);
}

static void printG(const char * str) {
  printInner(0, str);
}

#endif
