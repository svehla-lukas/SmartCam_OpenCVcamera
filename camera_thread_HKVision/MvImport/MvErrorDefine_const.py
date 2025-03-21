#!/usr/bin/env python
# -*- coding: utf-8 -*-

MV_OK                                        = 0x00000000  # <  Successed, no error

# 通用错误码定义:范围0x80000000-0x800000FF
MV_E_HANDLE                                  = 0x80000000  # <  Error or invalid handle
MV_E_SUPPORT                                 = 0x80000001  # <  Not supported function
MV_E_BUFOVER                                 = 0x80000002  # <  Buffer overflow
MV_E_CALLORDER                               = 0x80000003  # <  Function calling order error
MV_E_PARAMETER                               = 0x80000004  # <  Incorrect parameter
MV_E_RESOURCE                                = 0x80000006  # <  Applying resource failed
MV_E_NODATA                                  = 0x80000007  # <  No data
MV_E_PRECONDITION                            = 0x80000008  # <  Precondition error, or running environment changed
MV_E_VERSION                                 = 0x80000009  # <  Version mismatches
MV_E_NOENOUGH_BUF                            = 0x8000000A  # <  Insufficient memory
MV_E_ABNORMAL_IMAGE                          = 0x8000000B  # <  Abnormal image, maybe incomplete image because of lost packet
MV_E_LOAD_LIBRARY                            = 0x8000000C  # <  Load library failed
MV_E_NOOUTBUF                                = 0x8000000D  # <  No Avaliable Buffer
MV_E_UNKNOW                                  = 0x800000FF  # <  Unknown error

# GenICam系列错误:范围0x80000100-0x800001FF
MV_E_GC_GENERIC                              = 0x80000100  # <  General error
MV_E_GC_ARGUMENT                             = 0x80000101  # <  Illegal parameters
MV_E_GC_RANGE                                = 0x80000102  # <  The value is out of range
MV_E_GC_PROPERTY                             = 0x80000103  # <  Property
MV_E_GC_RUNTIME                              = 0x80000104  # <  Running environment error
MV_E_GC_LOGICAL                              = 0x80000105  # <  Logical error
MV_E_GC_ACCESS                               = 0x80000106  # <  Node accessing condition error
MV_E_GC_TIMEOUT                              = 0x80000107  # <  Timeout
MV_E_GC_DYNAMICCAST                          = 0x80000108  # <  Transformation exception
MV_E_GC_UNKNOW                               = 0x800001FF  # <  GenICam unknown error

# GigE_STATUS对应的错误码:范围0x80000200-0x800002FF
MV_E_NOT_IMPLEMENTED                         = 0x80000200  # <  The command is not supported by device
MV_E_INVALID_ADDRESS                         = 0x80000201  # <  The target address being accessed does not exist
MV_E_WRITE_PROTECT                           = 0x80000202  # <  The target address is not writable
MV_E_ACCESS_DENIED                           = 0x80000203  # <  No permission
MV_E_BUSY                                    = 0x80000204  # <  Device is busy, or network disconnected
MV_E_PACKET                                  = 0x80000205  # <  Network data packet error
MV_E_NETER                                   = 0x80000206  # <  Network error
MV_E_IP_CONFLICT                             = 0x80000221  # <  Device IP conflict

# USB_STATUS对应的错误码:范围0x80000300-0x800003FF
MV_E_USB_READ                                = 0x80000300  # <  Reading USB error
MV_E_USB_WRITE                               = 0x80000301  # <  Writing USB error
MV_E_USB_DEVICE                              = 0x80000302  # <  Device exception
MV_E_USB_GENICAM                             = 0x80000303  # <  GenICam error
MV_E_USB_BANDWIDTH                           = 0x80000304  # <  Insufficient bandwidth, this error code is newly added
MV_E_USB_DRIVER                              = 0x80000305  # <  Driver mismatch or unmounted drive
MV_E_USB_UNKNOW                              = 0x800003FF  # <  USB unknown error

# 升级时对应的错误码:范围0x80000400-0x800004FF
MV_E_UPG_FILE_MISMATCH                       = 0x80000400  # <  Firmware mismatches
MV_E_UPG_LANGUSGE_MISMATCH                   = 0x80000401  # <  Firmware language mismatches
MV_E_UPG_CONFLICT                            = 0x80000402  # <  Upgrading conflicted (repeated upgrading requests during device upgrade)
MV_E_UPG_INNER_ERR                           = 0x80000403  # <  Camera internal error during upgrade
MV_E_UPG_UNKNOW                              = 0x800004FF  # <  Unknown error during upgrade
