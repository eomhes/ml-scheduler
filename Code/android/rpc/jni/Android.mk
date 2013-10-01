
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := euclidean

LOCAL_C_INCLUDES := .

LOCAL_LDLIBS := -lz -lm

LOCAL_SRC_FILES := ../../../src/euclidean.c
#					../../../src/scheduling.c \
#				    ../../../src/and_rpc_clnt.c \
#                   ../../../src/tpl.c \
#                   ../../../src/ocl_utils.c \
#                   ../../../src/sobel.c

include $(BUILD_EXECUTABLE)
