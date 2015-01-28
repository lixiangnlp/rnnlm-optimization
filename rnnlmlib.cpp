///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3e
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//
///////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "rnnlmlib.h"
#include <sys/time.h>
#include <iostream>

using namespace rnn;

///// fast exp() implementation
static union{
    double d;
    struct{
        int j,i;
        } n;
} d2i;
#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i = EXP_A*(y)+(1072693248-EXP_C),d2i.d)

///// include blas
#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif
//

void printMatrix(synapse * i_matrix, size_t fromRows, size_t toRows, size_t fromColumns, size_t toColumns, size_t i_totalColumns)
{
    std::cout << std::endl;
    for (size_t i=fromRows; i<toRows; i++)
    {
        for (size_t j=fromColumns; j<toColumns; j++)
        {
            std::cout << i_matrix[j+i*i_totalColumns] << " ";
        }
        std::cout << std::endl;
    }
}

void printVector(neuron * i_vector, size_t from, size_t i_size)
{
    std::cout << std::endl;
    for (size_t i=from; i<from+i_size; i++)
    {
        std::cout << i_vector[i] << std::endl;
    }
}

//DenseMat&& convertToEigenMatrix(double *srcmatrix, int matrix_width, int from, int to, int from2, int to2)
//{
//    Eigen::Map<Eigen::MatrixXd, 1, Eigen::Stride<1, Eigen::Dynamic>> sourceMat(srcmatrix+from*matrix_width + from2, to-from, to2-from2, Eigen::Stride<1, Eigen::Dynamic>(1, matrix_width));
//    return std::move(DenseMat(sourceMat));
//}

//DenseVec&& convertToEigenVector(double *srcvec, int from2, int to2)
//{
//    Eigen::Map<Eigen::VectorXd> sourceVec(srcvec+from2, to2-from2);
//    return std::move(DenseVec(sourceVec));
//}

real CRnnLM::random(real min, real max)
{
    return rand()/(real)RAND_MAX*(max-min)+min;
}

void CRnnLM::setTrainFile(char *str)
{
    strcpy(train_file, str);
}

void CRnnLM::setValidFile(char *str)
{
    strcpy(valid_file, str);
}

void CRnnLM::setTestFile(char *str)
{
    strcpy(test_file, str);
}

void CRnnLM::setRnnLMFile(char *str)
{
    strcpy(rnnlm_file, str);
}


void CRnnLM::readWord(char *word, FILE *fin)
{
    int a=0, ch;

    while (!feof(fin)) {
    ch=fgetc(fin);

    if (ch==13) continue;

    if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
            if (a>0) {
                if (ch=='\n') ungetc(ch, fin);
                break;
            }

            if (ch=='\n') {
                strcpy(word, (char *)"</s>");
                return;
            }
            else continue;
        }

        word[a]=ch;
        a++;

        if (a>=MAX_STRING) {
            //printf("Too long word found!\n");   //truncate too long words
            a--;
        }
    }
    word[a]=0;
}


/* Version for bad corpora
void CRnnLM::readWord(char *word, FILE *fin) {
    int a = 0, ch, nextch;
    while (!feof(fin)) {
      ch=fgetc(fin);

      if (ch==13) continue;

      if ((ch=='-') || (ch==',') || (ch=='.')) {
        if (a==0) continue;
        nextch=fgetc(fin);
        if ((nextch==' ') || (nextch=='\t') || (nextch=='\n') || (nextch=='-') || (ch=='.') ) {
          ungetc(nextch, fin);
          break;
        }
        ungetc(nextch, fin);
      }

      if ((ch==' ') || (ch=='\t') || (ch=='\n') || (ch=='[') || (ch==']') || (ch=='!') || (ch=='"') ||
        (ch=='…') || (ch=='–') || (ch=='—') || (ch=='?') || (ch=='(') || (ch==')') || (ch==';') || (ch==':') || (ch=='»') || (ch=='«')) {
        if (a>0) {
          break;
        }
        else
          continue;
        }

        word[a]=ch;
        a++;

        if (a>=MAX_STRING) {
              //printf("Too long word found!\n");   //truncate too long words
          a--;
        }
      }
    word[a] = 0;
}
*/

int CRnnLM::getWordHash(char *word)
{
    unsigned int hash, a;

    hash=0;
    for (a=0; a<strlen(word); a++) hash=hash*237+word[a];
    hash=hash%vocab_hash_size;

    return hash;
}

int CRnnLM::searchVocab(char *word)
{
    int a;
    unsigned int hash;

    hash=getWordHash(word);

    if (vocab_hash[hash]==-1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];

    for (a=0; a<vocab_size; a++) {				//search in vocabulary
        if (!strcmp(word, vocab[a].word)) {
            vocab_hash[hash]=a;
            return a;
        }
    }

    return -1;							//return OOV if not found
}

int CRnnLM::readWordIndex(FILE *fin)
{
    char word[MAX_STRING];

    readWord(word, fin);
    if (feof(fin)) return -1;

    return searchVocab(word);
}

int CRnnLM::addWordToVocab(char *word)
{
    unsigned int hash;

    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn=0;
    vocab_size++;

    if (vocab_size+2>=vocab_max_size) {        //reallocate memory if needed
        vocab_max_size+=100;
        vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }

    hash=getWordHash(word);
    vocab_hash[hash]=vocab_size-1;

    return vocab_size-1;
}

void CRnnLM::sortVocab()
{
    int a, b, max;
    vocab_word swap;

    for (a=1; a<vocab_size; a++) {
        max=a;
        for (b=a+1; b<vocab_size; b++) if (vocab[max].cn<vocab[b].cn) max=b;

        swap=vocab[max];
        vocab[max]=vocab[a];
        vocab[a]=swap;
    }
}

void CRnnLM::learnVocabFromTrainFile()    //assumes that vocabulary is empty
{
    char word[MAX_STRING];
    FILE *fin;
    int a, i, train_wcn;

    for (a=0; a<vocab_hash_size; a++) vocab_hash[a]=-1;

    fin=fopen(train_file, "rb");

    vocab_size=0;

    addWordToVocab((char *)"</s>");

    train_wcn=0;
    while (1) {
        readWord(word, fin);
        if (feof(fin)) break;

        train_wcn++;

        i=searchVocab(word);
        if (i==-1) {
            a=addWordToVocab(word);
            vocab[a].cn=1;
        } else vocab[i].cn++;
    }

    sortVocab();

    //select vocabulary size
    /*a=0;
    while (a<vocab_size) {
    a++;
    if (vocab[a].cn==0) break;
    }
    vocab_size=a;*/

    if (debug_mode>0) {
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %d\n", train_wcn);
    }

    train_words=train_wcn;

    fclose(fin);
}

void CRnnLM::PrintVocabIntoFile(char* str) {
    char print_file[MAX_STRING];
    strcpy(print_file, str);
    FILE* fw = fopen(print_file, "w");
    fprintf (fw, "%d\n", vocab_size);
    for (int a = 0; a < vocab_size; ++a) {
        fprintf(fw, "%s %d\n", vocab[a].word, vocab[a].cn);
    }
    fclose(fw);
}

void CRnnLM::saveWeights()      //saves current weights and unit activations
{
    int a,b;

    for (a=0; a<layer0_size; a++) {
        neu0b_ac[a]=neu0_ac[a];
        neu0b_er[a]=neu0_er[a];
    }

    for (a=0; a<layer1_size; a++) {
        neu1b_ac[a]=neu1_ac[a];
        neu1b_er[a]=neu1_er[a];
    }

    for (a=0; a<layerc_size; a++) {
        neucb_ac[a]=neuc_ac[a];
        neucb_er[a]=neuc_er[a];
    }

    for (a=0; a<layer2_size; a++) {
        neu2b_ac[a]=neu2_ac[a];
        neu2b_er[a]=neu2_er[a];
    }

    for (b=0; b<layer1_size; b++) for (a=0; a<layer0_size; a++) {
    syn0b[a+b*layer0_size]=syn0[a+b*layer0_size];
    }

    if (layerc_size>0) {
    for (b=0; b<layerc_size; b++) for (a=0; a<layer1_size; a++) {
        syn1b[a+b*layer1_size]=syn1[a+b*layer1_size];
    }

    for (b=0; b<layer2_size; b++) for (a=0; a<layerc_size; a++) {
        syncb[a+b*layerc_size]=sync[a+b*layerc_size];
    }
    }
    else {
    for (b=0; b<layer2_size; b++) for (a=0; a<layer1_size; a++) {
        syn1b[a+b*layer1_size]=syn1[a+b*layer1_size];
    }
    }

    //for (a=0; a<direct_size; a++) syn_db[a]=syn_d[a];
}

void CRnnLM::restoreWeights()      //restores current weights and unit activations from backup copy
{
    int a,b;

    for (a=0; a<layer0_size; a++) {
        neu0_ac[a]=neu0b_ac[a];
        neu0_er[a]=neu0b_er[a];
    }

    for (a=0; a<layer1_size; a++) {
        neu1_ac[a]=neu1b_ac[a];
        neu1_er[a]=neu1b_er[a];
    }

    for (a=0; a<layerc_size; a++) {
        neuc_ac[a]=neucb_ac[a];
        neuc_er[a]=neucb_er[a];
    }

    for (a=0; a<layer2_size; a++) {
        neu2_ac[a]=neu2b_ac[a];
        neu2_er[a]=neu2b_er[a];
    }

    for (b=0; b<layer1_size; b++) for (a=0; a<layer0_size; a++) {
        syn0[a+b*layer0_size]=syn0b[a+b*layer0_size];
    }

    if (layerc_size>0) {
    for (b=0; b<layerc_size; b++) for (a=0; a<layer1_size; a++) {
        syn1[a+b*layer1_size]=syn1b[a+b*layer1_size];
    }

    for (b=0; b<layer2_size; b++) for (a=0; a<layerc_size; a++) {
        sync[a+b*layerc_size]=syncb[a+b*layerc_size];
    }
    }
    else {
    for (b=0; b<layer2_size; b++) for (a=0; a<layer1_size; a++) {
        syn1[a+b*layer1_size]=syn1b[a+b*layer1_size];
    }
    }

    //for (a=0; a<direct_size; a++) syn_d[a]=syn_db[a];
}

void CRnnLM::saveContext()		//useful for n-best list processing
{
    int a;

    for (a=0; a<layer1_size; a++) neu1b_ac[a]=neu1_ac[a];
}

void CRnnLM::restoreContext()
{
    int a;

    for (a=0; a<layer1_size; a++) neu1_ac[a]=neu1b_ac[a];
}

void CRnnLM::saveContext2()
{
    int a;

    for (a=0; a<layer1_size; a++) neu1b2_ac[a]=neu1_ac[a];
}

void CRnnLM::restoreContext2()
{
    int a;

    for (a=0; a<layer1_size; a++) neu1_ac[a]=neu1b2_ac[a];
}

void CRnnLM::initNet()
{
    int a, b, cl;

    layer0_size=vocab_size+layer1_size;
    layer2_size=vocab_size+class_size;

    neu0_ac=(neuron *)calloc(layer0_size, sizeof(neuron)); //input layer: <vocabulary[t]> and <hidden[t-1]>
    neu0_er=(neuron *)calloc(layer0_size, sizeof(neuron));
    neu1_ac=(neuron *)calloc(layer1_size, sizeof(neuron)); //hidden layer
    neu1_er=(neuron *)calloc(layer1_size, sizeof(neuron));
    neuc_ac=(neuron *)calloc(layerc_size, sizeof(neuron)); //compression layer - if we want to get compressed vocabulary vectors to be input for the hidden layer
    neuc_er=(neuron *)calloc(layerc_size, sizeof(neuron));
    neu2_ac=(neuron *)calloc(layer2_size, sizeof(neuron)); //output layer
    neu2_er=(neuron *)calloc(layer2_size, sizeof(neuron));

    syn0=(synapse *)calloc(layer0_size*layer1_size, sizeof(synapse)); //Tranform matrix: input layer -> hidden layer
    if (layerc_size==0)
    syn1=(synapse *)calloc(layer1_size*layer2_size, sizeof(synapse)); //Transform matrix: hidden layer -> output layer
    else {
    syn1=(synapse *)calloc(layer1_size*layerc_size, sizeof(synapse));
    sync=(synapse *)calloc(layerc_size*layer2_size, sizeof(synapse));
    }

    if (syn1==NULL) {
    printf("Memory allocation failed\n");
    exit(1);
    }

    if (layerc_size>0) if (sync==NULL) {
    printf("Memory allocation failed\n");
    exit(1);
    }

    syn_d=(direct_t *)calloc((long long)direct_size, sizeof(direct_t));

    if (syn_d==NULL) {
    printf("Memory allocation for direct connections failed (requested %lld bytes)\n", (long long)direct_size * (long long)sizeof(direct_t));
    exit(1);
    }

    neu0b_ac=( neuron *)calloc(layer0_size, sizeof( neuron)); //backup copies
    neu1b_ac=( neuron *)calloc(layer1_size, sizeof( neuron));
    neucb_ac=( neuron *)calloc(layerc_size, sizeof( neuron));
    neu1b2_ac=( neuron *)calloc(layer1_size, sizeof( neuron));
    neu2b_ac=( neuron *)calloc(layer2_size, sizeof( neuron));
    neu0b_er=( neuron *)calloc(layer0_size, sizeof( neuron)); //backup copies
    neu1b_er=( neuron *)calloc(layer1_size, sizeof( neuron));
    neucb_er=( neuron *)calloc(layerc_size, sizeof( neuron));
    neu1b2_er=( neuron *)calloc(layer1_size, sizeof( neuron));
    neu2b_er=( neuron *)calloc(layer2_size, sizeof( neuron));

    syn0b=(synapse *)calloc(layer0_size*layer1_size, sizeof(synapse));
    //syn1b=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
    if (layerc_size==0)
    syn1b=(synapse *)calloc(layer1_size*layer2_size, sizeof(synapse));
    else {
    syn1b=(synapse *)calloc(layer1_size*layerc_size, sizeof(synapse));
    syncb=(synapse *)calloc(layerc_size*layer2_size, sizeof(synapse));
    }

    if (syn1b==NULL) {
    printf("Memory allocation failed\n");
    exit(1);
    }

    for (a=0; a<layer0_size; a++) {
        neu0_ac[a]=0;
        neu0_er[a]=0;
    }

    for (a=0; a<layer1_size; a++) {
        neu1_ac[a]=0;
        neu1_er[a]=0;
    }

    for (a=0; a<layerc_size; a++) {
        neuc_ac[a]=0;
        neuc_er[a]=0;
    }

    for (a=0; a<layer2_size; a++) {
        neu2_ac[a]=0;
        neu2_er[a]=0;
    }

    for (b=0; b<layer1_size; b++) for (a=0; a<layer0_size; a++) {
        syn0[a+b*layer0_size]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1); // W{|H| * (|V|+|H|)}
    }

    if (layerc_size>0) {
    for (b=0; b<layerc_size; b++) for (a=0; a<layer1_size; a++) {
        syn1[a+b*layer1_size]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
    }

    for (b=0; b<layer2_size; b++) for (a=0; a<layerc_size; a++) {
        sync[a+b*layerc_size]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
    }
    }
    else {
    for (b=0; b<layer2_size; b++) for (a=0; a<layer1_size; a++) {
        syn1[a+b*layer1_size]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1); //W{(|V|+|C|) * |H|}
    }
    }

    long long aa;
    for (aa=0; aa<direct_size; aa++) syn_d[aa]=0;

    if (bptt>0) {
    bptt_history=(int *)calloc((bptt+bptt_block+10), sizeof(int));
    for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=-1;
    //
    bptt_hidden_ac=(neuron *)calloc((bptt+bptt_block+1)*layer1_size, sizeof(neuron));
    bptt_hidden_er=(neuron *)calloc((bptt+bptt_block+1)*layer1_size, sizeof(neuron));
    for (a=0; a<(bptt+bptt_block)*layer1_size; a++) {
        bptt_hidden_ac[a]=0;
        bptt_hidden_er[a]=0;
    }
    //
    bptt_syn0=(synapse *)calloc(layer0_size*layer1_size, sizeof(synapse));
    if (bptt_syn0==NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    }

    saveWeights();

    double df, dd;
    int i;

    df=0;
    dd=0;
    a=0;
    b=0;

    if (old_classes) {  	// old classes
        for (i=0; i<vocab_size; i++) b+=vocab[i].cn;
        for (i=0; i<vocab_size; i++) {
            df+=vocab[i].cn/(double)b;
            if (df>1) df=1;
            if (df>(a+1)/(double)class_size) {
                vocab[i].class_index=a;
                if (a<class_size-1) a++;
            } else {
                vocab[i].class_index=a;
            }
        }
    } else {			// new classes
        for (i=0; i<vocab_size; i++) b+=vocab[i].cn;
        for (i=0; i<vocab_size; i++) dd+=sqrt(vocab[i].cn/(double)b);
        for (i=0; i<vocab_size; i++) {
        df+=sqrt(vocab[i].cn/(double)b)/dd;
            if (df>1) df=1;
            if (df>(a+1)/(double)class_size) {
                vocab[i].class_index=a;
                if (a<class_size-1) a++;
            } else {
                vocab[i].class_index=a;
            }
    }
    }

    //allocate auxiliary class variables (for faster search when normalizing probability at output layer)

    class_words=(int **)calloc(class_size, sizeof(int *));
    class_cn=(int *)calloc(class_size, sizeof(int));
    class_max_cn=(int *)calloc(class_size, sizeof(int));

    for (i=0; i<class_size; i++) {
    class_cn[i]=0;
    class_max_cn[i]=10;
    class_words[i]=(int *)calloc(class_max_cn[i], sizeof(int));
    }

    for (i=0; i<vocab_size; i++) {
    cl=vocab[i].class_index;
    class_words[cl][class_cn[cl]]=i;
    class_cn[cl]++;
    if (class_cn[cl]+2>=class_max_cn[cl]) {
        class_max_cn[cl]+=10;
        class_words[cl]=(int *)realloc(class_words[cl], class_max_cn[cl]*sizeof(int));
    }
    }
}

void CRnnLM::saveNet()       //will save the whole network structure
{
    FILE *fo;
    int a, b;
    char str[1000];
    float fl;

    sprintf(str, "%s.temp", rnnlm_file);

    fo=fopen(str, "wb");
    if (fo==NULL) {
        printf("Cannot create file %s\n", rnnlm_file);
        exit(1);
    }
    fprintf(fo, "version: %d\n", version);
    fprintf(fo, "file format: %d\n\n", filetype);

    fprintf(fo, "training data file: %s\n", train_file);
    fprintf(fo, "validation data file: %s\n\n", valid_file);

    fprintf(fo, "last probability of validation data: %f\n", llogp);
    fprintf(fo, "number of finished iterations: %d\n", iter);

    fprintf(fo, "current position in training data: %d\n", train_cur_pos);
    fprintf(fo, "current probability of training data: %f\n", logp);
    fprintf(fo, "save after processing # words: %d\n", anti_k);
    fprintf(fo, "# of training words: %d\n", train_words);

    fprintf(fo, "input layer size: %d\n", layer0_size);
    fprintf(fo, "hidden layer size: %d\n", layer1_size);
    fprintf(fo, "compression layer size: %d\n", layerc_size);
    fprintf(fo, "output layer size: %d\n", layer2_size);

    fprintf(fo, "direct connections: %lld\n", direct_size);
    fprintf(fo, "direct order: %d\n", direct_order);

    fprintf(fo, "bptt: %d\n", bptt);
    fprintf(fo, "bptt block: %d\n", bptt_block);

    fprintf(fo, "vocabulary size: %d\n", vocab_size);
    fprintf(fo, "class size: %d\n", class_size);

    fprintf(fo, "old classes: %d\n", old_classes);
    fprintf(fo, "independent sentences mode: %d\n", independent);

    fprintf(fo, "starting learning rate: %f\n", starting_alpha);
    fprintf(fo, "current learning rate: %f\n", alpha);
    fprintf(fo, "learning rate decrease: %d\n", alpha_divide);
    fprintf(fo, "\n");

    fprintf(fo, "\nVocabulary:\n");
    for (a=0; a<vocab_size; a++) fprintf(fo, "%6d\t%10d\t%s\t%d\n", a, vocab[a].cn, vocab[a].word, vocab[a].class_index);


    if (filetype==TEXT) {
    fprintf(fo, "\nHidden layer activation:\n");
    for (a=0; a<layer1_size; a++) fprintf(fo, "%.4f\n", neu1_ac[a]);
    }
    if (filetype==BINARY) {
        for (a=0; a<layer1_size; a++) {
            fl=neu1_ac[a];
            fwrite(&fl, 4, 1, fo);
        }
    }
    //////////
    if (filetype==TEXT) {
    fprintf(fo, "\nWeights 0->1:\n");
    for (b=0; b<layer1_size; b++) {
            for (a=0; a<layer0_size; a++) {
            fprintf(fo, "%.4f\n", syn0[a+b*layer0_size]);
            }
    }
    }
    if (filetype==BINARY) {
    for (b=0; b<layer1_size; b++) {
            for (a=0; a<layer0_size; a++) {
            fl=syn0[a+b*layer0_size];
            fwrite(&fl, 4, 1, fo);
            }
    }
    }
    /////////
    if (filetype==TEXT) {
    if (layerc_size>0) {
        fprintf(fo, "\n\nWeights 1->c:\n");
        for (b=0; b<layerc_size; b++) {
        for (a=0; a<layer1_size; a++) {
                fprintf(fo, "%.4f\n", syn1[a+b*layer1_size]);
            }
            }

            fprintf(fo, "\n\nWeights c->2:\n");
        for (b=0; b<layer2_size; b++) {
        for (a=0; a<layerc_size; a++) {
                fprintf(fo, "%.4f\n", sync[a+b*layerc_size]);
            }
            }
    }
    else
    {
        fprintf(fo, "\n\nWeights 1->2:\n");
        for (b=0; b<layer2_size; b++) {
        for (a=0; a<layer1_size; a++) {
                fprintf(fo, "%.4f\n", syn1[a+b*layer1_size]);
            }
            }
        }
    }
    if (filetype==BINARY) {
    if (layerc_size>0) {
        for (b=0; b<layerc_size; b++) {
        for (a=0; a<layer1_size; a++) {
            fl=syn1[a+b*layer1_size];
                fwrite(&fl, 4, 1, fo);
            }
            }

        for (b=0; b<layer2_size; b++) {
        for (a=0; a<layerc_size; a++) {
                fl=sync[a+b*layerc_size];
                fwrite(&fl, 4, 1, fo);
            }
            }
    }
    else
    {
        for (b=0; b<layer2_size; b++) {
        for (a=0; a<layer1_size; a++) {
                fl=syn1[a+b*layer1_size];
                fwrite(&fl, 4, 1, fo);
            }
            }
        }
    }
    ////////
    if (filetype==TEXT) {
    fprintf(fo, "\nDirect connections:\n");
    long long aa;
    for (aa=0; aa<direct_size; aa++) {
            fprintf(fo, "%.2f\n", syn_d[aa]);
    }
    }
    if (filetype==BINARY) {
    long long aa;
    for (aa=0; aa<direct_size; aa++) {
            fl=syn_d[aa];
            fwrite(&fl, 4, 1, fo);

            /*fl=syn_d[aa]*4*256;			//saving direct connections this way will save 50% disk space; several times more compression is doable by clustering
            if (fl>(1<<15)-1) fl=(1<<15)-1;
            if (fl<-(1<<15)) fl=-(1<<15);
            si=(signed short int)fl;
            fwrite(&si, 2, 1, fo);*/
    }
    }
    ////////
    fclose(fo);

    rename(str, rnnlm_file);
}

void CRnnLM::goToDelimiter(int delim, FILE *fi)
{
    int ch=0;

    while (ch!=delim) {
        ch=fgetc(fi);
        if (feof(fi)) {
            printf("Unexpected end of file\n");
            exit(1);
        }
    }
}

void CRnnLM::restoreNet()    //will read whole network structure
{
    FILE *fi;
    int a, b, ver;
    float fl;
    char str[MAX_STRING];
    double d;

    fi=fopen(rnnlm_file, "rb");
    if (fi==NULL) {
    printf("ERROR: model file '%s' not found!\n", rnnlm_file);
    exit(1);
    }

    goToDelimiter(':', fi);
    fscanf(fi, "%d", &ver);
    if ((ver==4) && (version==5)) /* we will solve this later.. */ ; else
    if (ver!=version) {
        printf("Unknown version of file %s\n", rnnlm_file);
        exit(1);
    }
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &filetype);
    //
    goToDelimiter(':', fi);
    if (train_file_set==0) {
    fscanf(fi, "%s", train_file);
    } else fscanf(fi, "%s", str);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%s", valid_file);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &llogp);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &iter);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_cur_pos);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &logp);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &anti_k);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_words);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer0_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer1_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layerc_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer2_size);
    //
    if (ver>5) {
    goToDelimiter(':', fi);
    fscanf(fi, "%lld", &direct_size);
    }
    //
    if (ver>6) {
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &direct_order);
    }
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &bptt);
    //
    if (ver>4) {
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &bptt_block);
    } else bptt_block=10;
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &vocab_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &class_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &old_classes);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &independent);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &d);
    starting_alpha=d;
    //
    goToDelimiter(':', fi);
    if (alpha_set==0) {
    fscanf(fi, "%lf", &d);
    alpha=d;
    } else fscanf(fi, "%lf", &d);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &alpha_divide);
    //


    //read normal vocabulary
    if (vocab_max_size<vocab_size) {
    if (vocab!=NULL) free(vocab);
        vocab_max_size=vocab_size+1000;
        vocab=(struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));    //initialize memory for vocabulary
    }
    //
    goToDelimiter(':', fi);
    for (a=0; a<vocab_size; a++) {
    //fscanf(fi, "%d%d%s%d", &b, &vocab[a].cn, vocab[a].word, &vocab[a].class_index);
    fscanf(fi, "%d%d", &b, &vocab[a].cn);
    readWord(vocab[a].word, fi);
    fscanf(fi, "%d", &vocab[a].class_index);
    //printf("%d  %d  %s  %d\n", b, vocab[a].cn, vocab[a].word, vocab[a].class_index);
    }
    //
    if (neu0_ac==NULL) initNet();		//memory allocation here
    //


    if (filetype==TEXT) {
    goToDelimiter(':', fi);
    for (a=0; a<layer1_size; a++) {
        fscanf(fi, "%lf", &d);
        neu1_ac[a]=d;
    }
    }
    if (filetype==BINARY) {
    fgetc(fi);
    for (a=0; a<layer1_size; a++) {
        fread(&fl, 4, 1, fi);
        neu1_ac[a]=fl;
    }
    }
    //
    if (filetype==TEXT) {
    goToDelimiter(':', fi);
    for (b=0; b<layer1_size; b++) {
            for (a=0; a<layer0_size; a++) {
        fscanf(fi, "%lf", &d);
        syn0[a+b*layer0_size]=d;
        }
    }
    }
    if (filetype==BINARY) {
    for (b=0; b<layer1_size; b++) {
            for (a=0; a<layer0_size; a++) {
            fread(&fl, 4, 1, fi);
        syn0[a+b*layer0_size]=fl;
            }
    }
    }
    //
    if (filetype==TEXT) {
    goToDelimiter(':', fi);
    if (layerc_size==0) {	//no compress layer
        for (b=0; b<layer2_size; b++) {
            for (a=0; a<layer1_size; a++) {
            fscanf(fi, "%lf", &d);
            syn1[a+b*layer1_size]=d;
        }
            }
    }
    else
    {				//with compress layer
        for (b=0; b<layerc_size; b++) {
            for (a=0; a<layer1_size; a++) {
            fscanf(fi, "%lf", &d);
            syn1[a+b*layer1_size]=d;
        }
            }

            goToDelimiter(':', fi);

            for (b=0; b<layer2_size; b++) {
            for (a=0; a<layerc_size; a++) {
            fscanf(fi, "%lf", &d);
            sync[a+b*layerc_size]=d;
        }
            }
    }
    }
    if (filetype==BINARY) {
    if (layerc_size==0) {	//no compress layer
        for (b=0; b<layer2_size; b++) {
            for (a=0; a<layer1_size; a++) {
                fread(&fl, 4, 1, fi);
            syn1[a+b*layer1_size]=fl;
            }
            }
    }
    else
    {				//with compress layer
        for (b=0; b<layerc_size; b++) {
            for (a=0; a<layer1_size; a++) {
                fread(&fl, 4, 1, fi);
            syn1[a+b*layer1_size]=fl;
            }
            }

            for (b=0; b<layer2_size; b++) {
            for (a=0; a<layerc_size; a++) {
                fread(&fl, 4, 1, fi);
            sync[a+b*layerc_size]=fl;
            }
            }
    }
    }
    //
    if (filetype==TEXT) {
    goToDelimiter(':', fi);		//direct conenctions
    long long aa;
        for (aa=0; aa<direct_size; aa++) {
        fscanf(fi, "%lf", &d);
        syn_d[aa]=d;
    }
    }
    //
    if (filetype==BINARY) {
    long long aa;
        for (aa=0; aa<direct_size; aa++) {
            fread(&fl, 4, 1, fi);
        syn_d[aa]=fl;

        /*fread(&si, 2, 1, fi);
        fl=si/(float)(4*256);
        syn_d[aa]=fl;*/
        }
    }
    //

    saveWeights();

    fclose(fi);
}

void CRnnLM::netFlush()   //cleans all activations and error vectors
{
    int a;

    for (a=0; a<layer0_size-layer1_size; a++) {
        neu0_ac[a]=0;
        neu0_er[a]=0;
    }

    for (a=layer0_size-layer1_size; a<layer0_size; a++) {   //last hidden layer is initialized to vector of 0.1 values to prevent unstability
        neu0_ac[a]=0.1;
        neu0_er[a]=0;
    }

    for (a=0; a<layer1_size; a++) {
        neu1_ac[a]=0;
        neu1_er[a]=0;
    }

    for (a=0; a<layerc_size; a++) {
        neuc_ac[a]=0;
        neuc_er[a]=0;
    }

    for (a=0; a<layer2_size; a++) {
        neu2_ac[a]=0;
        neu2_er[a]=0;
    }
}

void CRnnLM::netReset()   //cleans hidden layer activation + bptt history
{
    int a, b;

    for (a=0; a<layer1_size; a++) {
        neu1_ac[a]=1.0;
    }

    copyHiddenLayerToInput();

    if (bptt>0) {
        for (a=1; a<bptt+bptt_block; a++) bptt_history[a]=0;
        for (a=bptt+bptt_block-1; a>1; a--) for (b=0; b<layer1_size; b++) {
            bptt_hidden_ac[a*layer1_size+b]=0;
            bptt_hidden_er[a*layer1_size+b]=0;
        }
    }

    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;
}

void CRnnLM::matrixXvector(double *dest, double *srcvec, double *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type)
{
        struct timeval start, end;

   //   Eigen::internal::set_is_malloc_allowed(false);
    if (type == 0)
    {
        Eigen::Map<Eigen::VectorXd> sourceVec(srcvec+from2, to2-from2);
        Eigen::Map<Eigen::MatrixXd, 1, Eigen::Stride<1, Eigen::Dynamic>> sourceMat(srcmatrix+from*matrix_width + from2, to-from, to2-from2, Eigen::Stride<1, Eigen::Dynamic>(1, matrix_width));
        Eigen::Map<Eigen::VectorXd> destVec(dest, to-from);

        DenseMat a(sourceMat);
        DenseVec b(sourceVec);
        gettimeofday(&start, NULL);
        DenseVec c = a * b;
//        std::cout << std::endl << sourceVec << std::endl;
       // std::cout << sourceMat << std::endl;
         //destVec.noalias() = sourceMat * sourceVec;
                        gettimeofday(&end, NULL);
                        destVec = c;

       // std::cout << std::endl << delta << std::endl;
                double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                             end.tv_usec - start.tv_usec) / 1.e6;

                        std::cout << std::endl << "Finished in " << delta << std::endl;
    }
    else
    {
        Eigen::Map<Eigen::RowVectorXd> sourceVec(srcvec+from, to-from);
        Eigen::Map<Eigen::MatrixXd, 1, Eigen::Stride<1, Eigen::Dynamic>> sourceMat(srcmatrix+from*matrix_width + from2, to-from, to2-from2, Eigen::Stride<1, Eigen::Dynamic>(1, matrix_width));
        Eigen::Map<Eigen::RowVectorXd> destVec(dest, to2-from2);
    //    std::cout << std::endl << sourceVec << std::endl;
     //   std::cout << sourceMat << std::endl;
                        gettimeofday(&start, NULL);
        destVec.noalias() += sourceMat.transpose() * sourceVec.transpose();
                         gettimeofday(&end, NULL);
       // sourceMat.transpose() * sourceVec.transpose();
      //  std::cout << std::endl << destVec << std::endl;
    }
   //  Eigen::internal::set_is_malloc_allowed(true);


}

void CRnnLM::computeNet(int last_word, int word)
{
    int a, b, c;
    real val;
    double sum;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here

    if (last_word!=-1) neu0_ac[last_word]=1;

    //propagate 0->1
    for (a=0; a<layer1_size; a++) neu1_ac[a]=0;
    for (a=0; a<layerc_size; a++) neuc_ac[a]=0;

//    printVector(neu0_ac, layer0_size - layer1_size, layer1_size + 1);

  //  printMatrix(syn0, 0, layer1_size, layer0_size - layer1_size, layer0_size, layer0_size);

    matrixXvector(neu1_ac, neu0_ac, syn0, layer0_size, 0, layer1_size, layer0_size-layer1_size, layer0_size, 0); // W[1:|H|, |V|+1:|V|+|H|] * s(t-1) : We use only the submatrix W' containing weights for the recurrent vector s(t-1)

//    printVector(neu1_ac,0,layer1_size);

    for (b=0; b<layer1_size; b++) {
        a=last_word;
        if (a!=-1) neu1_ac[b]+= neu0_ac[a] * syn0[a+b*layer0_size]; //adds projection of the w-th column of W to the corresponding cordinates of s(t)
    }

    //activate 1      --sigmoid
    for (a=0; a<layer1_size; a++) {
    if (neu1_ac[a]>50) neu1_ac[a]=50;  //for numerical stability
        if (neu1_ac[a]<-50) neu1_ac[a]=-50;  //for numerical stability
        val=-neu1_ac[a];
        neu1_ac[a]=1/(1+FAST_EXP(val)); //convert the hidden layer into activation levels
    }

    if (layerc_size>0) {
    matrixXvector(neuc_ac, neu1_ac, syn1, layer1_size, 0, layerc_size, 0, layer1_size, 0);
    //activate compression      --sigmoid
    for (a=0; a<layerc_size; a++) {
        if (neuc_ac[a]>50) neuc_ac[a]=50;  //for numerical stability
            if (neuc_ac[a]<-50) neuc_ac[a]=-50;  //for numerical stability
            val=-neuc_ac[a];
            neuc_ac[a]=1/(1+FAST_EXP(val));
    }
    }

    //1->2 class
    for (b=vocab_size; b<layer2_size; b++) neu2_ac[b]=0;

    if (layerc_size>0) {
    matrixXvector(neu2_ac, neuc_ac, sync, layerc_size, vocab_size, layer2_size, 0, layerc_size, 0);
    }
    else
    {
    matrixXvector(neu2_ac, neu1_ac, syn1, layer1_size, vocab_size, layer2_size, 0, layer1_size, 0);
    }

    //apply direct connections to classes
    if (direct_size>0) {
    unsigned long long hash[MAX_NGRAM_ORDER];	//this will hold pointers to syn_d that contains hash parameters

    for (a=0; a<direct_order; a++) hash[a]=0;

    for (a=0; a<direct_order; a++) {
        b=0;
        if (a>0) if (history[a-1]==-1) break;	//if OOV was in history, do not use this N-gram feature and higher orders
        hash[a]=PRIMES[0]*PRIMES[1];

        for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);	//update hash value based on words from the history
        hash[a]=hash[a]%(direct_size/2);		//make sure that starting hash index is in the first half of syn_d (second part is reserved for history->words features)
    }

    for (a=vocab_size; a<layer2_size; a++) {
        for (b=0; b<direct_order; b++) if (hash[b]) {
        neu2_ac[a]+=syn_d[hash[b]];		//apply current parameter and move to the next one
        hash[b]++;
        } else break;
    }
    }

    //activation 2   --softmax on classes
    sum=0;
    for (a=vocab_size; a<layer2_size; a++) {
    if (neu2_ac[a]>50) neu2_ac[a]=50;  //for numerical stability
    if (neu2_ac[a]<-50) neu2_ac[a]=-50;  //for numerical stability
        val=FAST_EXP(neu2_ac[a]);
        sum+=val;
        neu2_ac[a]=val;
    }
    for (a=vocab_size; a<layer2_size; a++) neu2_ac[a]/=sum;         //output layer activations now sum exactly to 1

    if (gen>0) return;	//if we generate words, we don't know what current word is -> only classes are estimated and word is selected in testGen()


    //1->2 word

    if (word!=-1) {
        for (c=0; c<class_cn[vocab[word].class_index]; c++) neu2_ac[class_words[vocab[word].class_index][c]]=0;
        if (layerc_size>0) {
        matrixXvector(neu2_ac, neuc_ac, sync, layerc_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layerc_size, 0);
    }
    else
    {
        matrixXvector(neu2_ac, neu1_ac, syn1, layer1_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layer1_size, 0);
    }
    }

    //apply direct connections to words
    if (word!=-1) if (direct_size>0) {
    unsigned long long hash[MAX_NGRAM_ORDER];

    for (a=0; a<direct_order; a++) hash[a]=0;

    for (a=0; a<direct_order; a++) {
        b=0;
        if (a>0) if (history[a-1]==-1) break;
        hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(vocab[word].class_index+1);

        for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
        hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
    }

    for (c=0; c<class_cn[vocab[word].class_index]; c++) {
        a=class_words[vocab[word].class_index][c];

        for (b=0; b<direct_order; b++) if (hash[b]) {
        neu2_ac[a]+=syn_d[hash[b]];
        hash[b]++;
        hash[b]=hash[b]%direct_size;
        } else break;
    }
    }

    //activation 2   --softmax on words
    sum=0;
    if (word!=-1) {
    for (c=0; c<class_cn[vocab[word].class_index]; c++) {
        a=class_words[vocab[word].class_index][c];
        if (neu2_ac[a]>50) neu2_ac[a]=50;  //for numerical stability
        if (neu2_ac[a]<-50) neu2_ac[a]=-50;  //for numerical stability
            val=FAST_EXP(neu2_ac[a]);
            sum+=val;
            neu2_ac[a]=val;
    }
    for (c=0; c<class_cn[vocab[word].class_index]; c++) neu2_ac[class_words[vocab[word].class_index][c]]/=sum;
    }
}

void CRnnLM::learnNet(int last_word, int word)
{
    int a, b, c, t, step;
    real beta2, beta3;

    beta2=beta*alpha;
    beta3=beta2*1;	//beta3 can be possibly larger than beta2, as that is useful on small datasets (if the final model is to be interpolated wich backoff model) - todo in the future

    if (word==-1) return;

    //compute error vectors
    for (c=0; c<class_cn[vocab[word].class_index]; c++) {
    a=class_words[vocab[word].class_index][c];
        neu2_er[a]=(0-neu2_ac[a]);
    }
    neu2_er[word]=(1-neu2_ac[word]);	//word part

    //flush error
    for (a=0; a<layer1_size; a++) neu1_er[a]=0;
    for (a=0; a<layerc_size; a++) neuc_er[a]=0;

    for (a=vocab_size; a<layer2_size; a++) {
        neu2_er[a]=(0-neu2_ac[a]);
    }
    neu2_er[vocab[word].class_index+vocab_size]=(1-neu2_ac[vocab[word].class_index+vocab_size]);	//class part

    //
    if (direct_size>0) {	//learn direct connections between words
    if (word!=-1) {
        unsigned long long hash[MAX_NGRAM_ORDER];

        for (a=0; a<direct_order; a++) hash[a]=0;

        for (a=0; a<direct_order; a++) {
        b=0;
        if (a>0) if (history[a-1]==-1) break;
        hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(vocab[word].class_index+1);

            for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
        hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
        }

        for (c=0; c<class_cn[vocab[word].class_index]; c++) {
        a=class_words[vocab[word].class_index][c];

        for (b=0; b<direct_order; b++) if (hash[b]) {
            syn_d[hash[b]]+=alpha*neu2_er[a] - syn_d[hash[b]]*beta3;
            hash[b]++;
            hash[b]=hash[b]%direct_size;
        } else break;
        }
    }
    }
    //
    //learn direct connections to classes
    if (direct_size>0) {	//learn direct connections between words and classes
    unsigned long long hash[MAX_NGRAM_ORDER];

    for (a=0; a<direct_order; a++) hash[a]=0;

    for (a=0; a<direct_order; a++) {
        b=0;
        if (a>0) if (history[a-1]==-1) break;
        hash[a]=PRIMES[0]*PRIMES[1];

        for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
        hash[a]=hash[a]%(direct_size/2);
    }

    for (a=vocab_size; a<layer2_size; a++) {
        for (b=0; b<direct_order; b++) if (hash[b]) {
        syn_d[hash[b]]+=alpha*neu2_er[a] - syn_d[hash[b]]*beta3;
        hash[b]++;
        } else break;
    }
    }
    //


    if (layerc_size>0) {
    matrixXvector(neuc_er, neu2_er, sync, layerc_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layerc_size, 1);

    t=class_words[vocab[word].class_index][0]*layerc_size;
    for (c=0; c<class_cn[vocab[word].class_index]; c++) {
        b=class_words[vocab[word].class_index][c];
        if ((counter%10)==0)	//regularization is done every 10. step
        for (a=0; a<layerc_size; a++) sync[a+t]+=alpha*neu2_er[b]*neuc_ac[a] - sync[a+t]*beta2;
        else
        for (a=0; a<layerc_size; a++) sync[a+t]+=alpha*neu2_er[b]*neuc_ac[a];
        t+=layerc_size;
    }
    //
    matrixXvector(neuc_er, neu2_er, sync, layerc_size, vocab_size, layer2_size, 0, layerc_size, 1);		//propagates errors 2->c for classes

    c=vocab_size*layerc_size;
    for (b=vocab_size; b<layer2_size; b++) {
        if ((counter%10)==0) {	//regularization is done every 10. step
        for (a=0; a<layerc_size; a++) sync[a+c]+=alpha*neu2_er[b]*neuc_ac[a] - sync[a+c]*beta2;	//weight c->2 update
        }
        else {
            for (a=0; a<layerc_size; a++) sync[a+c]+=alpha*neu2_er[b]*neuc_ac[a];	//weight c->2 update
        }
        c+=layerc_size;
    }

    for (a=0; a<layerc_size; a++) neuc_er[a]=neuc_er[a]*neuc_ac[a]*(1-neuc_ac[a]);    //error derivation at compression layer

    ////

    matrixXvector(neu1_er, neuc_er, syn1, layer1_size, 0, layerc_size, 0, layer1_size, 1);		//propagates errors c->1

    for (b=0; b<layerc_size; b++) {
        for (a=0; a<layer1_size; a++) syn1[a+b*layer1_size]+=alpha*neuc_er[b]*neu1_ac[a];	//weight 1->c update
    }
    }
    else
    {
        matrixXvector(neu1_er, neu2_er, syn1, layer1_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layer1_size, 1);

 //       printVector(neu1_er,0,layer1_size);

        t=class_words[vocab[word].class_index][0]*layer1_size;
    for (c=0; c<class_cn[vocab[word].class_index]; c++) {
        b=class_words[vocab[word].class_index][c];
        if ((counter%10)==0)	//regularization is done every 10. step
        for (a=0; a<layer1_size; a++) syn1[a+t]+=alpha*neu2_er[b]*neu1_ac[a] - syn1[a+t]*beta2;
        else
        for (a=0; a<layer1_size; a++) syn1[a+t]+=alpha*neu2_er[b]*neu1_ac[a];
        t+=layer1_size;
    }
    matrixXvector(neu1_er, neu2_er, syn1, layer1_size, vocab_size, layer2_size, 0, layer1_size, 1);		//propagates errors 2->1 for classes
//    printVector(neu1_er, 0, layer1_size);
    c=vocab_size*layer1_size;
    for (b=vocab_size; b<layer2_size; b++) {
        if ((counter%10)==0) {	//regularization is done every 10. step
        for (a=0; a<layer1_size; a++) syn1[a+c]+=alpha*neu2_er[b]*neu1_ac[a] - syn1[a+c]*beta2;	//weight 1->2 update
        }
        else {
            for (a=0; a<layer1_size; a++) syn1[a+c]+=alpha*neu2_er[b]*neu1_ac[a];	//weight 1->2 update
        }
        c+=layer1_size;
    }
    }

//    printMatrix(syn1,0,layer2_size, 0,layer1_size,layer1_size);

    //

    ///////////////
 //   printVector(neu1_ac, 0, layer1_size);
 //   printVector(neu1_er, 0, layer1_size);
    if (bptt<=1) {		//bptt==1 -> normal BP
    for (a=0; a<layer1_size; a++) neu1_er[a]=neu1_er[a]*neu1_ac[a]*(1-neu1_ac[a]);    //error derivation at layer 1

    //weight update 1->0
    a=last_word;
    if (a!=-1) {
        if ((counter%10)==0)
        for (b=0; b<layer1_size; b++) syn0[a+b*layer0_size]+=alpha*neu1_er[b]*neu0_ac[a] - syn0[a+b*layer0_size]*beta2;
        else
        for (b=0; b<layer1_size; b++) syn0[a+b*layer0_size]+=alpha*neu1_er[b]*neu0_ac[a];
    }

    if ((counter%10)==0) {
        for (b=0; b<layer1_size; b++) for (a=layer0_size-layer1_size; a<layer0_size; a++) syn0[a+b*layer0_size]+=alpha*neu1_er[b]*neu0_ac[a] - syn0[a+b*layer0_size]*beta2;
    }
    else {
        for (b=0; b<layer1_size; b++) for (a=layer0_size-layer1_size; a<layer0_size; a++) syn0[a+b*layer0_size]+=alpha*neu1_er[b]*neu0_ac[a];
    }
//    printMatrix(syn0, 0, layer1_size, layer0_size - layer1_size, layer0_size, layer0_size);
    }
    else		//BPTT
    {
    for (b=0; b<layer1_size; b++) bptt_hidden_ac[b]=neu1_ac[b];
    for (b=0; b<layer1_size; b++) bptt_hidden_er[b]=neu1_er[b];

    if (((counter%bptt_block)==0) || (independent && (word==0))) {
        for (step=0; step<bptt+bptt_block-2; step++) {
        for (a=0; a<layer1_size; a++) neu1_er[a]=neu1_er[a]*neu1_ac[a]*(1-neu1_ac[a]);    //error derivation at layer 1

        //weight update 1->0
        a=bptt_history[step];
        if (a!=-1)
        for (b=0; b<layer1_size; b++) {
                bptt_syn0[a+b*layer0_size]+=alpha*neu1_er[b];//*neu0[a].ac; --should be always set to 1
        }

        for (a=layer0_size-layer1_size; a<layer0_size; a++) neu0_er[a]=0;

        matrixXvector(neu0_er, neu1_er, syn0, layer0_size, 0, layer1_size, layer0_size-layer1_size, layer0_size, 1);		//propagates errors 1->0
        for (b=0; b<layer1_size; b++) for (a=layer0_size-layer1_size; a<layer0_size; a++) {
            //neu0_er[a] += neu1_er[b] * syn0[a+b*layer0_size];
                bptt_syn0[a+b*layer0_size]+=alpha*neu1_er[b]*neu0_ac[a];
        }

        for (a=0; a<layer1_size; a++) {		//propagate error from time T-n to T-n-1
                neu1_er[a]=neu0_er[a+layer0_size-layer1_size]+ bptt_hidden_er[(step+1)*layer1_size+a];
        }

        if (step<bptt+bptt_block-3)
        for (a=0; a<layer1_size; a++) {
            neu1_ac[a]=bptt_hidden_ac[(step+1)*layer1_size+a];
            neu0_ac[a+layer0_size-layer1_size]=bptt_hidden_ac[(step+2)*layer1_size+a];
        }
        }

        for (a=0; a<(bptt+bptt_block)*layer1_size; a++) {
        bptt_hidden_er[a]=0;
        }


        for (b=0; b<layer1_size; b++) neu1_ac[b]=bptt_hidden_ac[b];		//restore hidden layer after bptt


        //
        for (b=0; b<layer1_size; b++) {		//copy temporary syn0
        if ((counter%10)==0) {
            for (a=layer0_size-layer1_size; a<layer0_size; a++) {
                    syn0[a+b*layer0_size]+=bptt_syn0[a+b*layer0_size] - syn0[a+b*layer0_size]*beta2;
                bptt_syn0[a+b*layer0_size]=0;
                }
        }
        else {
            for (a=layer0_size-layer1_size; a<layer0_size; a++) {
                syn0[a+b*layer0_size]+=bptt_syn0[a+b*layer0_size];
                bptt_syn0[a+b*layer0_size]=0;
                }
        }

        if ((counter%10)==0) {
            for (step=0; step<bptt+bptt_block-2; step++) if (bptt_history[step]!=-1) {
                syn0[bptt_history[step]+b*layer0_size]+=bptt_syn0[bptt_history[step]+b*layer0_size] - syn0[bptt_history[step]+b*layer0_size]*beta2;
                bptt_syn0[bptt_history[step]+b*layer0_size]=0;
                }
        }
        else {
            for (step=0; step<bptt+bptt_block-2; step++) if (bptt_history[step]!=-1) {
                syn0[bptt_history[step]+b*layer0_size]+=bptt_syn0[bptt_history[step]+b*layer0_size];
            bptt_syn0[bptt_history[step]+b*layer0_size]=0;
            }
        }
        }
    }
    }
}

void CRnnLM::copyHiddenLayerToInput()
{
    int a;

    for (a=0; a<layer1_size; a++) {
        neu0_ac[a+layer0_size-layer1_size]=neu1_ac[a];
    }
}

void CRnnLM::trainNet()
{
    int a, b, word, last_word, wordcn;
    char log_name[200];
    FILE *fi, *flog;
    clock_t start, now;

    sprintf(log_name, "%s.output.txt", rnnlm_file);

    printf("Starting training using file %s\n", train_file);
    starting_alpha=alpha;

    fi=fopen(rnnlm_file, "rb");
    if (fi!=NULL) {
    fclose(fi);
    printf("Restoring network from file to continue training...\n");
    restoreNet();
    } else {
    learnVocabFromTrainFile();
    initNet();
    iter=0;
    }

    if (class_size>vocab_size) {
    printf("WARNING: number of classes exceeds vocabulary size!\n");
    }

    counter=train_cur_pos;

    //saveNet();

    while (1) {
        printf("Iter: %3d\tAlpha: %f\t   ", iter, alpha);
        fflush(stdout);

        if (bptt>0) for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=0;
        for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;

        //TRAINING PHASE
        netFlush();

        fi=fopen(train_file, "rb");
        last_word=0;

        if (counter>0) for (a=0; a<counter; a++) word=readWordIndex(fi);	//this will skip words that were already learned if the training was interrupted

        start=clock();

        while (1) {
            counter++;

            if ((counter%10000)==0) if ((debug_mode>1)) {
            now=clock();
            if (train_words>0)
                printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %.2f%%   Words/sec: %.1f ", 13, iter, alpha, -logp/log10(2)/counter, counter/(real)train_words*100, counter/((double)(now-start)/1000000.0));
            else
                printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %dK", 13, iter, alpha, -logp/log10(2)/counter, counter/1000);
            fflush(stdout);
            }

            if ((anti_k>0) && ((counter%anti_k)==0)) {
            train_cur_pos=counter;
            saveNet();
            }

        word=readWordIndex(fi);     //read next word
            computeNet(last_word, word);      //compute probability distribution
            if (feof(fi)) break;        //end of file: test on validation data, iterate till convergence

            if (word!=-1) logp+=log10(neu2_ac[vocab[word].class_index+vocab_size] * neu2_ac[word]);

            if ((logp!=logp) || (isinf(logp))) {
                printf("\nNumerical error %d %f %f\n", word, neu2_ac[word], neu2_ac[vocab[word].class_index+vocab_size]);
                exit(1);
            }

            //
            if (bptt>0) {		//shift memory needed for bptt to next time step
        for (a=bptt+bptt_block-1; a>0; a--) bptt_history[a]=bptt_history[a-1];
        bptt_history[0]=last_word;

        for (a=bptt+bptt_block-1; a>0; a--) for (b=0; b<layer1_size; b++) {
            bptt_hidden_ac[a*layer1_size+b]=bptt_hidden_ac[(a-1)*layer1_size+b];
            bptt_hidden_er[a*layer1_size+b]=bptt_hidden_er[(a-1)*layer1_size+b];
        }
            }
            //
            learnNet(last_word, word);

            copyHiddenLayerToInput();

            if (last_word!=-1) neu0_ac[last_word]=0;  //delete previous activation

            last_word=word;

            for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
            history[0]=last_word;

        if (independent && (word==0)) netReset();
        }
        fclose(fi);

    now=clock();
        printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Words/sec: %.1f   ", 13, iter, alpha, -logp/log10(2)/counter, counter/((double)(now-start)/1000000.0));

        if (one_iter==1) {	//no validation data are needed and network is always saved with modified weights
            printf("\n");
        logp=0;
            saveNet();
            break;
        }

        //VALIDATION PHASE
        netFlush();

        fi=fopen(valid_file, "rb");
    if (fi==NULL) {
        printf("Valid file not found\n");
        exit(1);
    }

        flog=fopen(log_name, "ab");
    if (flog==NULL) {
        printf("Cannot open log file\n");
        exit(1);
    }

        //fprintf(flog, "Index   P(NET)          Word\n");
        //fprintf(flog, "----------------------------------\n");

        last_word=0;
        logp=0;
        wordcn=0;
        while (1) {
            word=readWordIndex(fi);     //read next word
            computeNet(last_word, word);      //compute probability distribution
            if (feof(fi)) break;        //end of file: report LOGP, PPL

            if (word!=-1) {
            logp+=log10(neu2_ac[vocab[word].class_index+vocab_size] * neu2_ac[word]);
            wordcn++;
            }

            /*if (word!=-1)
                fprintf(flog, "%d\t%f\t%s\n", word, neu2[word].ac, vocab[word].word);
            else
                fprintf(flog, "-1\t0\t\tOOV\n");*/

            //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
            copyHiddenLayerToInput();

            if (last_word!=-1) neu0_ac[last_word]=0;  //delete previous activation

            last_word=word;

            for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
            history[0]=last_word;

        if (independent && (word==0)) netReset();
        }
        fclose(fi);

        fprintf(flog, "\niter: %d\n", iter);
        fprintf(flog, "valid log probability: %f\n", logp);
        fprintf(flog, "PPL net: %f\n", exp10(-logp/(real)wordcn));

        fclose(flog);

        printf("VALID entropy: %.4f\n", -logp/log10(2)/wordcn);

        counter=0;
    train_cur_pos=0;

        if (logp<llogp)
            restoreWeights();
        else
            saveWeights();

        if (logp*min_improvement<llogp) {
            if (alpha_divide==0) alpha_divide=1;
            else {
                saveNet();
                break;
            }
        }

        if (alpha_divide) alpha/=2;

        llogp=logp;
        logp=0;
        iter++;
        saveNet();
    }
}

void CRnnLM::testNet()
{
    int a, b, word, last_word, wordcn;
    FILE *fi, *flog, *lmprob=NULL;
    real prob_other, log_other, log_combine;
    double d;

    restoreNet();

    if (use_lmprob) {
    lmprob=fopen(lmprob_file, "rb");
    }

    //TEST PHASE
    //netFlush();

    fi=fopen(test_file, "rb");
    //sprintf(str, "%s.%s.output.txt", rnnlm_file, test_file);
    //flog=fopen(str, "wb");
    flog=stdout;

    if (debug_mode>1)	{
    if (use_lmprob) {
            fprintf(flog, "Index   P(NET)          P(LM)           Word\n");
            fprintf(flog, "--------------------------------------------------\n");
    } else {
            fprintf(flog, "Index   P(NET)          Word\n");
            fprintf(flog, "----------------------------------\n");
    }
    }

    last_word=0;					//last word = end of sentence
    logp=0;
    log_other=0;
    log_combine=0;
    prob_other=0;
    wordcn=0;
    copyHiddenLayerToInput();

    if (bptt>0) for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=0;
    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;
    if (independent) netReset();

    while (1) {

        word=readWordIndex(fi);		//read next word
        computeNet(last_word, word);		//compute probability distribution
        if (feof(fi)) break;		//end of file: report LOGP, PPL

        if (use_lmprob) {
            fscanf(lmprob, "%lf", &d);
            prob_other=d;

            goToDelimiter('\n', lmprob);
        }

        if ((word!=-1) || (prob_other>0)) {
            if (word==-1) {
            logp+=-8;		//some ad hoc penalty - when mixing different vocabularies, single model score is not real PPL
            log_combine+=log10(0 * lambda + prob_other*(1-lambda));
            } else {
            logp+=log10(neu2_ac[vocab[word].class_index+vocab_size] * neu2_ac[word]);
            log_combine+=log10(neu2_ac[vocab[word].class_index+vocab_size] * neu2_ac[word]*lambda + prob_other*(1-lambda));
            }
            log_other+=log10(prob_other);
            wordcn++;
        }

    if (debug_mode>1) {
            if (use_lmprob) {
            if (word!=-1) fprintf(flog, "%d\t%.10f\t%.10f\t%s", word, neu2_ac[vocab[word].class_index+vocab_size] *neu2_ac[word], prob_other, vocab[word].word);
            else fprintf(flog, "-1\t0\t\t0\t\tOOV");
            } else {
            if (word!=-1) fprintf(flog, "%d\t%.10f\t%s", word, neu2_ac[vocab[word].class_index+vocab_size] *neu2_ac[word], vocab[word].word);
            else fprintf(flog, "-1\t0\t\tOOV");
            }

            fprintf(flog, "\n");
        }

        if (dynamic>0) {
            if (bptt>0) {
                for (a=bptt+bptt_block-1; a>0; a--) bptt_history[a]=bptt_history[a-1];
                bptt_history[0]=last_word;

                for (a=bptt+bptt_block-1; a>0; a--) for (b=0; b<layer1_size; b++) {
                    bptt_hidden_ac[a*layer1_size+b]=bptt_hidden_ac[(a-1)*layer1_size+b];
                    bptt_hidden_er[a*layer1_size+b]=bptt_hidden_er[(a-1)*layer1_size+b];
            }
            }
            //
            alpha=dynamic;
            learnNet(last_word, word);    //dynamic update
        }
        copyHiddenLayerToInput();

        if (last_word!=-1) neu0_ac[last_word]=0;  //delete previous activation

        last_word=word;

        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
        history[0]=last_word;

    if (independent && (word==0)) netReset();
    }
    fclose(fi);
    if (use_lmprob) fclose(lmprob);

    //write to log file
    if (debug_mode>0) {
    fprintf(flog, "\ntest log probability: %f\n", logp);
    if (use_lmprob) {
            fprintf(flog, "test log probability given by other lm: %f\n", log_other);
            fprintf(flog, "test log probability %f*rnn + %f*other_lm: %f\n", lambda, 1-lambda, log_combine);
    }

    fprintf(flog, "\nPPL net: %f\n", exp10(-logp/(real)wordcn));
    if (use_lmprob) {
            fprintf(flog, "PPL other: %f\n", exp10(-log_other/(real)wordcn));
            fprintf(flog, "PPL combine: %f\n", exp10(-log_combine/(real)wordcn));
    }
    }

    fclose(flog);
}

void CRnnLM::testNbest()
{
    int a, word, last_word, wordcn;
    FILE *fi, *flog, *lmprob=NULL;
    float prob_other; //has to be float so that %f works in fscanf
    real log_other, log_combine, senp;
    //int nbest=-1;
    int nbest_cn=0;
    char ut1[MAX_STRING], ut2[MAX_STRING];

    restoreNet();
    computeNet(0, 0);
    copyHiddenLayerToInput();
    saveContext();
    saveContext2();

    if (use_lmprob) {
    lmprob=fopen(lmprob_file, "rb");
    } else lambda=1;		//!!! for simpler implementation later

    //TEST PHASE
    //netFlush();

    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;

    if (!strcmp(test_file, "-")) fi=stdin; else fi=fopen(test_file, "rb");

    //sprintf(str, "%s.%s.output.txt", rnnlm_file, test_file);
    //flog=fopen(str, "wb");
    flog=stdout;

    last_word=0;		//last word = end of sentence
    logp=0;
    log_other=0;
    prob_other=0;
    log_combine=0;
    wordcn=0;
    senp=0;
    strcpy(ut1, (char *)"");
    while (1) {
    if (last_word==0) {
        fscanf(fi, "%s", ut2);

        if (nbest_cn==1) saveContext2();		//save context after processing first sentence in nbest

        if (strcmp(ut1, ut2)) {
        strcpy(ut1, ut2);
        nbest_cn=0;
        restoreContext2();
        saveContext();
        } else restoreContext();

        nbest_cn++;

        copyHiddenLayerToInput();
        }


    word=readWordIndex(fi);     //read next word
    if (lambda>0) computeNet(last_word, word);      //compute probability distribution
        if (feof(fi)) break;        //end of file: report LOGP, PPL


        if (use_lmprob) {
            fscanf(lmprob, "%f", &prob_other);
            goToDelimiter('\n', lmprob);
        }

        if (word!=-1)
        neu2_ac[word]*=neu2_ac[vocab[word].class_index+vocab_size];

        if (word!=-1) {
            logp+=log10(neu2_ac[word]);

            log_other+=log10(prob_other);

            log_combine+=log10(neu2_ac[word]*lambda + prob_other*(1-lambda));

            senp+=log10(neu2_ac[word]*lambda + prob_other*(1-lambda));

            wordcn++;
        } else {
            //assign to OOVs some score to correctly rescore nbest lists, reasonable value can be less than 1/|V| or backoff LM score (in case it is trained on more data)
            //this means that PPL results from nbest list rescoring are not true probabilities anymore (as in open vocabulary LMs)

            real oov_penalty=-5;	//log penalty

            if (prob_other!=0) {
            logp+=log10(prob_other);
            log_other+=log10(prob_other);
            log_combine+=log10(prob_other);
            senp+=log10(prob_other);
            } else {
            logp+=oov_penalty;
            log_other+=oov_penalty;
            log_combine+=oov_penalty;
            senp+=oov_penalty;
            }
            wordcn++;
        }

        //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
        copyHiddenLayerToInput();

        if (last_word!=-1) neu0_ac[last_word]=0;  //delete previous activation

        if (word==0) {		//write last sentence log probability / likelihood
            fprintf(flog, "%f\n", senp);
            senp=0;
    }

        last_word=word;

        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
        history[0]=last_word;

    if (independent && (word==0)) netReset();
    }
    fclose(fi);
    if (use_lmprob) fclose(lmprob);

    if (debug_mode>0) {
    printf("\ntest log probability: %f\n", logp);
    if (use_lmprob) {
            printf("test log probability given by other lm: %f\n", log_other);
            printf("test log probability %f*rnn + %f*other_lm: %f\n", lambda, 1-lambda, log_combine);
    }

    printf("\nPPL net: %f\n", exp10(-logp/(real)wordcn));
    if (use_lmprob) {
            printf("PPL other: %f\n", exp10(-log_other/(real)wordcn));
            printf("PPL combine: %f\n", exp10(-log_combine/(real)wordcn));
    }
    }

    fclose(flog);
}

void CRnnLM::testGen()
{
    int i, word, cla, last_word, wordcn, c, b, a=0;
    real f, g, sum, val;

    restoreNet();

    word=0;
    last_word=0;					//last word = end of sentence
    wordcn=0;
    copyHiddenLayerToInput();
    while (wordcn<gen) {
        computeNet(last_word, 0);		//compute probability distribution

        f=random(0, 1);
        g=0;
        i=vocab_size;
        while ((g<f) && (i<layer2_size)) {
            g+=neu2_ac[i];
            i++;
        }
        cla=i-1-vocab_size;

        if (cla>class_size-1) cla=class_size-1;
        if (cla<0) cla=0;

        //
        // !!!!!!!!  THIS WILL WORK ONLY IF CLASSES ARE CONTINUALLY DEFINED IN VOCAB !!! (like class 10 = words 11 12 13; not 11 12 16)  !!!!!!!!
        // forward pass 1->2 for words
        for (c=0; c<class_cn[cla]; c++) neu2_ac[class_words[cla][c]]=0;
        matrixXvector(neu2_ac, neu1_ac, syn1, layer1_size, class_words[cla][0], class_words[cla][0]+class_cn[cla], 0, layer1_size, 0);

        //apply direct connections to words
    if (word!=-1) if (direct_size>0) {
            unsigned long long hash[MAX_NGRAM_ORDER];

            for (a=0; a<direct_order; a++) hash[a]=0;

            for (a=0; a<direct_order; a++) {
                b=0;
                if (a>0) if (history[a-1]==-1) break;
                hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(cla+1);

                for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
                hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
            }

            for (c=0; c<class_cn[cla]; c++) {
            a=class_words[cla][c];

            for (b=0; b<direct_order; b++) if (hash[b]) {
                neu2_ac[a]+=syn_d[hash[b]];
                    hash[b]++;
                hash[b]=hash[b]%direct_size;
                } else break;
            }
    }

        //activation 2   --softmax on words
    sum=0;
        for (c=0; c<class_cn[cla]; c++) {
            a=class_words[cla][c];
            if (neu2_ac[a]>50) neu2_ac[a]=50;  //for numerical stability
            if (neu2_ac[a]<-50) neu2_ac[a]=-50;  //for numerical stability
            val=FAST_EXP(neu2_ac[a]);
            sum+=val;
            neu2_ac[a]=val;
        }
        for (c=0; c<class_cn[cla]; c++) neu2_ac[class_words[cla][c]]/=sum;
    //

    f=random(0, 1);
        g=0;
        /*i=0;
        while ((g<f) && (i<vocab_size)) {
            g+=neu2[i].ac;
            i++;
        }*/
        for (c=0; c<class_cn[cla]; c++) {
            a=class_words[cla][c];
            g+=neu2_ac[a];
            if (g>f) break;
        }
        word=a;

    if (word>vocab_size-1) word=vocab_size-1;
        if (word<0) word=0;

    //printf("%s %d %d\n", vocab[word].word, cla, word);
    if (word!=0)
        printf("%s ", vocab[word].word);
    else
        printf("\n");

        copyHiddenLayerToInput();

        if (last_word!=-1) neu0_ac[last_word]=0;  //delete previous activation

        last_word=word;

        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
        history[0]=last_word;

    if (independent && (word==0)) netReset();

        wordcn++;
    }
}
