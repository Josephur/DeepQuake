/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Stand-in for <SDL_opengl.h>: the GL types the vendored headers mention.
 *
 * Since #42 vendored the client side, quakedef.h pulls in the real
 * glquake.h, whose externs need these. The simulator never calls GL, so
 * only the types must exist; the client build uses real SDL2's header
 * instead of this file.
 */
#ifndef DQSIM_SHIM_SDL_OPENGL_H
#define DQSIM_SHIM_SDL_OPENGL_H

typedef unsigned int GLuint;
typedef int GLint;
typedef unsigned int GLenum;
typedef float GLfloat;
typedef int GLsizei;
typedef unsigned char GLboolean;
typedef ptrdiff_t GLsizeiptrARB;
typedef ptrdiff_t GLintptrARB;
typedef char GLcharARB;
typedef char GLchar;
typedef void GLvoid;

#ifndef APIENTRYP
#define APIENTRYP *
#endif

typedef void (APIENTRYP PFNGLMULTITEXCOORD2FARBPROC) (GLenum target, GLfloat s, GLfloat t);
typedef void (APIENTRYP PFNGLACTIVETEXTUREARBPROC) (GLenum texture);
typedef void (APIENTRYP PFNGLCLIENTACTIVETEXTUREARBPROC) (GLenum texture);
typedef void (APIENTRYP PFNGLBINDBUFFERARBPROC) (GLenum target, GLuint buffer);
typedef void (APIENTRYP PFNGLDELETEBUFFERSARBPROC) (GLsizei n, const GLuint *buffers);
typedef void (APIENTRYP PFNGLGENBUFFERSARBPROC) (GLsizei n, GLuint *buffers);
typedef void (APIENTRYP PFNGLBUFFERDATAARBPROC) (GLenum target, GLsizeiptrARB size,
                                                 const GLvoid *data);
typedef void (APIENTRYP PFNGLBUFFERSUBDATAARBPROC) (GLenum target, GLintptrARB offset,
                                                    GLsizeiptrARB size, const GLvoid *data);
typedef void (APIENTRYP PFNGLATTACHSHADERPROC) (GLuint program, GLuint shader);
typedef void (APIENTRYP PFNGLBINDATTRIBLOCATIONPROC) (GLuint program, GLuint index,
                                                      const GLcharARB *name);
typedef void (APIENTRYP PFNGLCOMPILESHADERPROC) (GLuint shader);
typedef GLuint (APIENTRYP PFNGLCREATEPROGRAMPROC) (void);
typedef GLuint (APIENTRYP PFNGLCREATESHADERPROC) (GLenum type);
typedef void (APIENTRYP PFNGLDELETEPROGRAMPROC) (GLuint program);
typedef void (APIENTRYP PFNGLDELETESHADERPROC) (GLuint shader);
typedef void (APIENTRYP PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint index);
typedef void (APIENTRYP PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint index);
typedef GLint (APIENTRYP PFNGLGETATTRIBLOCATIONPROC) (GLuint program, const GLcharARB *name);
typedef void (APIENTRYP PFNGLGETPROGRAMINFOLOGPROC) (GLuint program, GLsizei bufSize,
                                                     GLsizei *length, GLcharARB *infoLog);
typedef void (APIENTRYP PFNGLGETPROGRAMIVPROC) (GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRYP PFNGLGETSHADERINFOLOGPROC) (GLuint shader, GLsizei bufSize,
                                                    GLsizei *length, GLcharARB *infoLog);
typedef void (APIENTRYP PFNGLGETSHADERIVPROC) (GLuint shader, GLenum pname, GLint *params);
typedef GLint (APIENTRYP PFNGLGETUNIFORMLOCATIONPROC) (GLuint program, const GLcharARB *name);
typedef void (APIENTRYP PFNGLLINKPROGRAMPROC) (GLuint program);
typedef void (APIENTRYP PFNGLSHADERSOURCEPROC) (GLuint shader, GLsizei count,
                                                const GLcharARB **strings,
                                                const GLint *lengths);
typedef void (APIENTRYP PFNGLUNIFORM1FPROC) (GLint location, GLfloat v0);
typedef void (APIENTRYP PFNGLUNIFORM1IPROC) (GLint location, GLint v0);
typedef void (APIENTRYP PFNGLUNIFORM3FPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
typedef void (APIENTRYP PFNGLUNIFORM4FPROC) (GLint location, GLfloat v0, GLfloat v1,
                                             GLfloat v2, GLfloat v3);
typedef void (APIENTRYP PFNGLUSEPROGRAMPROC) (GLuint program);
typedef void (APIENTRYP PFNGLVERTEXATTRIBPOINTERPROC) (GLuint index, GLint size, GLenum type,
                                                       GLboolean normalized, GLsizei stride,
                                                       const GLvoid *pointer);

#endif
