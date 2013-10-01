
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := sobelfilter
LOCAL_SRC_FILES := SobelFilterHost.cpp
include $(BUILD_EXECUTABLE)
