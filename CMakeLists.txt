# Object Cadenza
#
# Copyright (c) 2014 Samsung Electronics Co., Ltd. All rights reserved.
#
# Authors: Andrey Moiseenko <moiseenko.a@samsung.com>,
#          Alexander Romanenko <Alexander Romanenko>,
#          Irina Piontkovskaya <p.irina@samsung.com>,
#          Manh Ha Le <mh.le@samsung.com>,
#          Mikhail Kudinov <m.kudinov@samsung.com>,
#          Stepan Muzychka <s.muzychka@samsung.com>

# root

cmake_minimum_required (VERSION 2.8)

# Project declaration should be the first statement, before that some variables are undefined
project(Cadenza)

# Define target platform - system and bitness.
# For now, use such hack for determinint Android cross-compilation. This method is cpecific
# to the current custom toolchain-file. The fila was taken from here:
# https://github.com/taka-no-me/android-cmake.git
if (DEFINED ANDROID_ABI)
    set(TARGET_SYSTEM "Android")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(TARGET_SYSTEM "Linux")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(TARGET_SYSTEM "Windows")
else()
    set(TARGET_SYSTEM "Unknown")
endif()

message("Target system is: ${TARGET_SYSTEM}")

set(bitness 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(bitness 64)
endif()

# Define linkage type.
# For now shared linkage could be enabled via "cmake -DLINK_SHARED=true ..." command
if (LINK_SHARED)
    set(LIBRARY_TYPE SHARED)
else()
    set(LIBRARY_TYPE STATIC)
endif()

if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(SEND_ERROR "In-source builds are not allowed.")
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

#set(CMAKE_BUILD_TYPE Debug)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(CONFIG_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/config)
#set(CONFIG_ROOT_DIR config)

if (MSVC)
#CSV MANAGER
    add_definitions(-DCSV_MANAGER_CONFIG_PATH="${CONFIG_ROOT_DIR}/CSVManager/Config_Windows/")
#CRF
    add_definitions(-DCRF_TESTING_MODEL_OUTPUT_PATH="${CONFIG_ROOT_DIR}/src/Config_Windows/Model_Output.txt")
    add_definitions(-DCRF_CONFIG_FOLDER="${CONFIG_ROOT_DIR}/CRF/Config_Windows/")
    add_definitions(-DCRF_TEST_SET_PATH="${CONFIG_ROOT_DIR}/CRF/Config_Windows/TestSet.txt")
    add_definitions(-DCRF_TRAINING_SET_PATH="${CONFIG_ROOT_DIR}/CRF/Config_Windows/TrainingSet.txt")
    add_definitions(-DCRF_ALL_TAGS_FILE_PATH="${CONFIG_ROOT_DIR}/CRF/Config_Windows/AllTags.txt")
# DICTIONARY
    add_definitions(-DBIN_DIC_DATA_PATH="${CONFIG_ROOT_DIR}/Dictionary/Data/")
#DISAMBIGUATOR
    add_definitions(-DDISAMBIGUATOR_CONFIG_FOLDER="${CONFIG_ROOT_DIR}/Disambiguator/Config_Windows/")
    add_definitions(-DDISAMBIGUATOR_MODEL_OUTPUT_PATH="${CONFIG_ROOT_DIR}/Disambiguator/Config_Windows/Model_Output.txt")
    add_definitions(-DDISAMBIGUATOR_TEST_SET_PATH="${CONFIG_ROOT_DIR}/Disambiguator/Config_Windows/TestSet.txt")
    add_definitions(-DDISAMBIGUATOR_TRAINING_SET_PATH="${CONFIG_ROOT_DIR}/Disambiguator/Config_Windows/TrainingSet.txt")
    add_definitions(-DDISAMBIGUATOR_EXCEPTION_FILE_PATH="${CONFIG_ROOT_DIR}/Disambiguator/Config_Windows/FileTranslationsForFeatureCalculator.txt")
    add_definitions(-DDISAMBIGUATOR_RESULT_TESTING_FILE="${CONFIG_ROOT_DIR}/Disambiguator/Config_Windows/Token_Label.txt")
    add_definitions(-DDISAMBIGUATOR_POSSIBLE_LABELS_FILE="${CONFIG_ROOT_DIR}/Disambiguator/Config_Windows/Possible labels.txt")
    add_definitions(-DDISAMBIGUATOR_RELEARN_SET_FILE="${CONFIG_ROOT_DIR}/Disambiguator/Config_Windows/RelearnSet.txt")
#GRAPHVIZ
    add_definitions(-DGRAPHVIZ_PATH="${CONFIG_ROOT_DIR}/../Graphviz/")
#NAME_ENTITY_RECOGNIZER
    add_definitions(-DNE_RECOGNIZER_CONFIG_PATH="${CONFIG_ROOT_DIR}/NER/Config_Windows/")
# TEXTCLASSIFICATIONEN
    add_definitions(-DTEXT_CLASSIFICATION_EN_DATA_PATH="${CONFIG_ROOT_DIR}/TextClassificationEn/Data/")
#TOKENIZER
    add_definitions(-DTOKENIZER_CONFIG_FOLDER="${CONFIG_ROOT_DIR}/Tokenizer/Config_Windows/")
#SENTENCE SPLITTER
    add_definitions(-DSSPLITTER_CONFIG="${CONFIG_ROOT_DIR}/SentenceSplitter/Config/sent_split_model")
# STRESSER
    add_definitions(-DSTRESSER_CONFIG_FOLDER="${CONFIG_ROOT_DIR}/Stresser/Config_Windows/")
#SYNTAX
    add_definitions(-DSYNTAX_CONFIG_FOLDER="${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Windows/")
    add_definitions(-DSYNTAX_DRAW_TREE_SCRIPT="${PROJECT_ROOT_DIR}/SyntaxTreeBuilder/drawTree.bat")
    add_definitions(-DSYNTAX_MODEL_OUTPUT_PATH="${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Windows/Model_Output.txt")
    add_definitions(-DSYNTAX_OUTPUT_TREE_PATH="${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/tree.dot")
    add_definitions(-DSYNTAX_TEST_SET_PATH="${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Windows/TestSet.txt")
    add_definitions(-DSYNTAX_TRAINING_SET_PATH="${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Windows/TrainingSet.txt")
# MORPHOLOGY SYNTAX UNION
    add_definitions(-DMORPHOLOGY_SYNTAX_UNION_CONFIG_FOLDER="${CONFIG_ROOT_DIR}/MorphologySyntaxUnion/Config_Windows/")
    add_definitions(-DCOMPOUND_DISAMBIGUATOR_MODEL_OUTPUT_PATH="${CONFIG_ROOT_DIR}/MorphologySyntaxUnion/Config_Windows/Model_Output.txt")
    add_definitions(-DMORPHOLOGY_SYNTAX_UNION_TEST_SET_PATH="${CONFIG_ROOT_DIR}/MorphologySyntaxUnion/Config_Windows/TestSet.txt")
    add_definitions(-DMORPHOLOGY_SYNTAX_UNION_TRAINING_SET_PATH="${CONFIG_ROOT_DIR}/MorphologySyntaxUnion/Config_Windows/TrainingSet.txt")
#STAT CALC
    add_definitions(-DSTAT_CALC_PREPROC_RULES="${CONFIG_ROOT_DIR}/StatCalc/preprocessing_rules.txt")
    add_definitions(-DSSPLITTER_CONFIG="${CONFIG_ROOT_DIR}/SentenceSplitter/Config/sent_split_model")
# SERVER
    add_definitions(-DSERVER_PATH=\"/data/del/\")
#NEWS_READER_DEMO
    add_definitions(-DNEWS_READER_DEMO_DATA=\"/media/smaug/NewsReaderDemo/Data/\")
#NEWS_READER_DEMO_ENG
    add_definitions(-DNEWS_READER_DEMO_ENG_DATA=\"/media/smaug/NewsReaderDemo-eng/Data/\")
#TEXTSUMMARIZATION
    add_definitions(-DTEXT_SUMMARIZATION_DATA=\"/media/smaug/TextSummarization/\")
#TEXTSUMMARIZATION
    add_definitions(-DTEXT_SUMMARIZATION_ENG_DATA=\"/media/smaug/TextSummarization-eng/\")
elseif (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

#CSV MANAGER
    add_definitions(-DCSV_MANAGER_CONFIG_PATH=\"${CONFIG_ROOT_DIR}/CSVManager/Config_Linux/\")
#CRF
    add_definitions(-DCRF_TESTING_MODEL_OUTPUT_PATH=\"${CONFIG_ROOT_DIR}/CRF/Config_Linux/Model_Output.txt\")
    add_definitions(-DCRF_CONFIG_FOLDER=\"${CONFIG_ROOT_DIR}/CRF/Config_Linux/\")
    add_definitions(-DCRF_TEST_SET_PATH=\"${CONFIG_ROOT_DIR}/CRF/Config_Linux/TestSet.txt\")
    add_definitions(-DCRF_TRAINING_SET_PATH=\"${CONFIG_ROOT_DIR}/CRF/Config_Linux/TrainingSet.txt\")
    add_definitions(-DCRF_ALL_TAGS_FILE_PATH=\"${CONFIG_ROOT_DIR}/CRF/Config_Linux/AllTags.txt\")
# DICTIONARY
    add_definitions(-DBIN_DIC_DATA_PATH=\"${CONFIG_ROOT_DIR}/Dictionary/Data/\")
#DISAMBIGUATOR
    add_definitions(-DDISAMBIGUATOR_CONFIG_FOLDER=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/\")
    add_definitions(-DDISAMBIGUATOR_MODEL_OUTPUT_PATH=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/Model_Output.txt\")
    add_definitions(-DDISAMBIGUATOR_TEST_SET_PATH=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/TestSet.txt\")
    add_definitions(-DDISAMBIGUATOR_TRAINING_SET_PATH=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/TrainingSet.txt\")
    add_definitions(-DDISAMBIGUATOR_EXCEPTION_FILE_PATH=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/FileTranslationsForFeatureCalculator.txt\")
    add_definitions(-DDISAMBIGUATOR_RESULT_TESTING_FILE=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/Token_Label.txt\")
    add_definitions(-DDISAMBIGUATOR_POSSIBLE_LABELS_FILE=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/Possible_labels.txt\")
    add_definitions(-DDISAMBIGUATOR_RELEARN_SET_FILE=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/RelearnSet.txt\")
    add_definitions(-DDISAMBIGUATOR_RELEARN_INPUT_SET_FILE=\"${CONFIG_ROOT_DIR}/Disambiguator/Config_Linux/RelearnInputSet.txt\")
# GRAPHVIZ
    add_definitions(-DGRAPHVIZ_PATH=\"\")
# KINOPOISK DEMO
    add_definitions(-DKINOPOISK_DEMO_CONFIG_FOLDER=\"${CONFIG_ROOT_DIR}/KinopoiskDemo/Config_Linux/\")
# NAME_ENTITY_RECOGNIZER
    add_definitions(-DNE_RECOGNIZER_CONFIG_PATH=\"${CONFIG_ROOT_DIR}/NER/Config_Linux/\")
# NLP FILE PROCESSOR
    add_definitions(-DNLP_FILE_PROCESSOR_CONFIG_FOLDER=\"${CONFIG_ROOT_DIR}/NLPFileProcessor/Config_Linux/\")
# TEXTCLASSIFICATIONEN
    add_definitions(-DTEXT_CLASSIFICATION_EN_DATA_PATH=\"${CONFIG_ROOT_DIR}/TextClassificationEn/Data/\")
# TOKENIZER
    add_definitions(-DTOKENIZER_CONFIG_FOLDER=\"${CONFIG_ROOT_DIR}/Tokenizer/Config_Linux/\")
# SENTENCE SPLITTER
    add_definitions(-DSSPLITTER_CONFIG=\"${CONFIG_ROOT_DIR}/SentenceSplitter/Config/sent_split_model\")
# STRESSER
    add_definitions(-DSTRESSER_CONFIG_FOLDER=\"${CONFIG_ROOT_DIR}/Stresser/Config_Linux/\")
#SYNTAX
    add_definitions(-DSYNTAX_CONFIG_FOLDER=\"${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Linux/\")
    add_definitions(-DSYNTAX_DRAW_TREE_SCRIPT=\"${PROJECT_ROOT_DIR}/SyntaxTreeBuilder/drawTree.sh\")
    add_definitions(-DSYNTAX_MODEL_OUTPUT_PATH=\"${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Linux/Model_Output.txt\")
    add_definitions(-DSYNTAX_OUTPUT_TREE_PATH=\"${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/tree.dot\")
    add_definitions(-DSYNTAX_TEST_SET_PATH=\"${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Linux/TestSet.txt\")
    add_definitions(-DSYNTAX_TRAINING_SET_PATH=\"${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Linux/TrainingSet.txt\")
    add_definitions(-DSYNTAX_EDGE_CALCULATOR_CONFIG_PATH=\"${CONFIG_ROOT_DIR}/SyntaxTreeBuilder/Config_Linux/EdgeCalculatorConfig.txt\")
#SYNTAX_SUMMARY
    add_definitions(-DSYNTAX_SUMMARY_TOKEN_IMPORTANCE_MODEL=\"${CONFIG_ROOT_DIR}/SyntaxSummary/Config_Linux/token-importance-model.txt\")
    add_definitions(-DSYNTAX_SUMMARY_TALK_TOKENS_FILE=\"${CONFIG_ROOT_DIR}/SyntaxSummary/Config_Linux/talk-tokens.txt\")
    add_definitions(-DSYNTAX_SUMMARY_SRC_FOLDER=\"${PROJECT_ROOT_DIR}/SyntaxSummary/\")
# MORPHOLOGY SYNTAX UNION
    add_definitions(-DMORPHOLOGY_SYNTAX_UNION_CONFIG_FOLDER=\"${CONFIG_ROOT_DIR}/MorphologySyntaxUnion/Config_Linux/\")
    add_definitions(-DCOMPOUND_DISAMBIGUATOR_MODEL_OUTPUT_PATH=\"${CONFIG_ROOT_DIR}/MorphologySyntaxUnion/Config_Linux/Model_Output.txt\")
    add_definitions(-DMORPHOLOGY_SYNTAX_UNION_TEST_SET_PATH=\"${CONFIG_ROOT_DIR}/MorphologySyntaxUnion/Config_Linux/TestSet.txt\")
    add_definitions(-DMORPHOLOGY_SYNTAX_UNION_TRAINING_SET_PATH=\"${CONFIG_ROOT_DIR}/MorphologySyntaxUnion/Config_Linux/TrainingSet.txt\")
# STAT CALC
    add_definitions(-DSTAT_CALC_PREPROC_RULES=\"${CONFIG_ROOT_DIR}/StatCalc/preprocessing_rules.txt\")
# SERVER
    add_definitions(-DSERVER_PATH=\"/media/smaug/\")
#NEWS_READER_DEMO
    add_definitions(-DNEWS_READER_DEMO_DATA=\"/media/smaug/NewsReaderDemo/Data/\")
#NEWS_READER_DEMO_ENG
    add_definitions(-DNEWS_READER_DEMO_ENG_DATA=\"/media/smaug/NewsReaderDemo-eng2/Data/\")
#TEXT_SUMMARIZATION
    add_definitions(-DTEXT_SUMMARIZATION_DATA=\"/media/smaug/TextSummarization/\")
#TEXT_SUMMARIZATION_ENG
    add_definitions(-DTEXT_SUMMARIZATION_ENG_DATA=\"/media/smaug/TextSummarization-eng/\")
#TOOLS
    add_definitions(-DTOOLS_SRC_FOLDER=\"${PROJECT_ROOT_DIR}/Tools/\")
endif()

if (MSVC)
    # Force to always compile with W3
    set(wlevel "/W3")
    if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "${wlevel}" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wlevel}")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi /Ob2")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
    add_definitions(-DMSVC)

    include_directories(${PROJECT_SOURCE_DIR}/include_for_msvc)
elseif (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

    # Update if necessary
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -c ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11") #TODO:  -Wconversion -Weffc++")


if (BUILD_MODE)
    if(${BUILD_MODE} STREQUAL DEBUG)
        message("Building DEBUG...")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb3 -Wno-unknown-pragmas")
    elseif(${BUILD_MODE} STREQUAL RELEASE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mtune=native -march=native -O3")
        message("Building RELEASE...")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mtune=native -march=native -ffast-math -O3")
       # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb3 -Wno-unknown-pragmas")
        message("Building DEFAULT...")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb3 -Wno-unknown-pragmas")
    message("Building DEFAULT...")
endif()

endif()

add_definitions(-D__STDC_LIMIT_MACROS)

if (ENABLE_COVERAGE)
    FIND_PROGRAM( GCOV_PATH gcov )
    FIND_PROGRAM( LCOV_PATH lcov )
    FIND_PROGRAM( GENHTML_PATH genhtml )
    FIND_PROGRAM( GCOVR_PATH gcovr)
    if (NOT GCOV_PATH)
        message(WARNING "gcov not found, coverage report is disabled")
        set(ENABLE_COVERAGE FALSE)
    endif()
endif()

if (ENABLE_COVERAGE)
    add_definitions(-fprofile-arcs)
    add_definitions(-ftest-coverage)

    file(WRITE "${CMAKE_BINARY_DIR}/coverage.sh"
        "cd ${CMAKE_BINARY_DIR}\n"
    )

    if (GENHTML_PATH)
        file(APPEND "${CMAKE_BINARY_DIR}/coverage.sh"
            "if [ \"$1\" == \"refresh\" ] ; then\n"
            "    ${LCOV_PATH} --directory . --zerocounters\n"
            "    make test\n"
            "fi\n"

            "${LCOV_PATH} --directory . --capture --output-file coverage.info\n"
            "${LCOV_PATH} --remove coverage.info 'test/*' '/usr/*' '*/Poco/*' '*/sqlite3.c' '*/Loki/*' '*/loki/*' --output-file coverage.info.clean\n"
            "${GENHTML_PATH} -o lcov coverage.info.clean\n"
        )
    endif()

    if (GCOVR_PATH)
        file(APPEND "${CMAKE_BINARY_DIR}/coverage.sh"
            "${GCOVR_PATH} -x -r .. -e '.*(build|test|sqlite3\\.c|Poco|[Ll]oki).*' -o gcovr.xml\n"
        )
    endif()
endif()
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)



# Find VLD
# VLD is a Windows-only memory leak detector
if ((${TARGET_SYSTEM} STREQUAL "Windows") AND (EXISTS "${CMAKE_CURRENT_LIST_DIR}/vld_path.cmake"))
    include("${CMAKE_CURRENT_LIST_DIR}/vld_path.cmake")
endif()

add_subdirectory(src bin)
add_subdirectory(test test)

# write build command to file
# useful for automatic building
build_command(bldcmd)
file(WRITE "${CMAKE_BINARY_DIR}/build.cmd"
    "cd ${CMAKE_BINARY_DIR}\n"
    ${bldcmd})
