option(DQSIM_SANITIZE "Build our code and tests with ASan and UBSan" OFF)

# The engine source to compile. Leave at the default except to build an
# instrumented scratch copy for debugging (see "Instrumenting the engine" in
# AGENTS.md); never point a dev or release build elsewhere.
set(DQSIM_QUAKESPASM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/quakespasm CACHE PATH
    "Directory holding the QuakeSpasm source (its Quake/ subdirectory is compiled)")
set(QS_DIR ${DQSIM_QUAKESPASM_DIR}/Quake)

# Include order matters: src/shim provides stand-ins for SDL.h, SDL_opengl.h
# and glquake.h, which are deliberately not vendored.
set(DQSIM_ENGINE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/src/shim ${QS_DIR})

# The vendored QuakeSpasm server core, compiled unmodified. See
# third_party/quakespasm/UPSTREAM.md for the file list and its origin.
add_library(quakespasm_server OBJECT
  ${QS_DIR}/cmd.c
  ${QS_DIR}/common.c
  ${QS_DIR}/crc.c
  ${QS_DIR}/cvar.c
  ${QS_DIR}/gl_model.c
  ${QS_DIR}/host.c
  ${QS_DIR}/host_cmd.c
  ${QS_DIR}/mathlib.c
  ${QS_DIR}/pr_cmds.c
  ${QS_DIR}/pr_edict.c
  ${QS_DIR}/pr_exec.c
  ${QS_DIR}/strlcat.c
  ${QS_DIR}/strlcpy.c
  ${QS_DIR}/sv_main.c
  ${QS_DIR}/sv_move.c
  ${QS_DIR}/sv_user.c
  ${QS_DIR}/world.c
  ${QS_DIR}/zone.c
)
target_include_directories(quakespasm_server PRIVATE ${DQSIM_ENGINE_INCLUDES})
target_compile_definitions(quakespasm_server PRIVATE
  USE_SDL2
  # Route every rand() call in the engine to our seeded generator (src/rng.c).
  rand=dqsim_rand
)
# Upstream code is not ours to make warning-clean; it is built with GNU
# extensions and without warnings. -fno-strict-aliasing matches upstream.
set_target_properties(quakespasm_server PROPERTIES C_EXTENSIONS ON)
target_compile_options(quakespasm_server PRIVATE -w -fno-strict-aliasing)

# sv_phys.c is the one vendored file compiled differently (#31): every
# QuakeC program execution it initiates (thinks, touches, the player and
# frame thinks) is routed to the brain hook in src/dqsim.c, which passes
# all but the nominated monster's decision thinks straight through to the
# real PR_ExecuteProgram. The vendored file itself is unmodified.
add_library(quakespasm_phys OBJECT ${QS_DIR}/sv_phys.c)
target_include_directories(quakespasm_phys PRIVATE ${DQSIM_ENGINE_INCLUDES})
target_compile_definitions(quakespasm_phys PRIVATE
  USE_SDL2
  rand=dqsim_rand
  PR_ExecuteProgram=dqsim_hook_execute_program
)
set_target_properties(quakespasm_phys PROPERTIES C_EXTENSIONS ON)
target_compile_options(quakespasm_phys PRIVATE -w -fno-strict-aliasing)

add_library(dqsim SHARED
  src/client_stubs.c
  src/dqsim.c
  src/net_fake.c
  src/platform.c
  src/rng.c
  $<TARGET_OBJECTS:quakespasm_server>
  $<TARGET_OBJECTS:quakespasm_phys>
)
set_target_properties(dqsim PROPERTIES C_EXTENSIONS ON)
target_include_directories(dqsim PUBLIC include PRIVATE ${DQSIM_ENGINE_INCLUDES})
target_compile_definitions(dqsim PRIVATE
  DQSIM_BUILDING
  USE_SDL2
  DQSIM_VERSION_STRING="${PROJECT_VERSION}"
)
# cvar_t is initialised upstream-style with trailing fields omitted.
target_compile_options(dqsim PRIVATE -Wall -Wextra -Werror -Wno-missing-field-initializers -fno-strict-aliasing)
target_link_libraries(dqsim PRIVATE m)
# Fail the build if any engine symbol is left undefined.
target_link_options(dqsim PRIVATE -Wl,-z,defs)

add_executable(dqsim_bench tools/dqsim_bench.c)
target_link_libraries(dqsim_bench PRIVATE dqsim)
target_compile_options(dqsim_bench PRIVATE -Wall -Wextra -Werror)

if(BUILD_TESTING)
  # Directory containing id1/ for tests that need game data. Those tests
  # report "skipped" (exit code 77) when it has no id1/pak0.pak.
  set(DQSIM_TEST_GAMEDATA "${PROJECT_SOURCE_DIR}/gamedata" CACHE PATH
      "Game data directory used by engine tests")

  # Sanitizers go on the test programs only: a sanitized shared library
  # cannot be loaded from an unsanitized Python interpreter.
  function(dqsim_add_test name)
    add_executable(${name} tests/${name}.c)
    target_link_libraries(${name} PRIVATE dqsim m)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Werror)
    if(DQSIM_SANITIZE)
      target_compile_options(${name} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${name} PRIVATE -fsanitize=address,undefined)
    endif()
  endfunction()

  dqsim_add_test(test_version)
  add_test(NAME dqsim.version COMMAND test_version ${PROJECT_VERSION})

  dqsim_add_test(test_sim)
  add_test(NAME dqsim.sim COMMAND test_sim ${DQSIM_TEST_GAMEDATA})
  set_tests_properties(dqsim.sim PROPERTIES SKIP_RETURN_CODE 77 LABELS gamedata)

  dqsim_add_test(test_snapshot)
  add_test(NAME dqsim.snapshot COMMAND test_snapshot ${DQSIM_TEST_GAMEDATA})
  set_tests_properties(dqsim.snapshot PROPERTIES SKIP_RETURN_CODE 77 LABELS gamedata)

  dqsim_add_test(test_hook)
  add_test(NAME dqsim.hook COMMAND test_hook ${DQSIM_TEST_GAMEDATA})
  set_tests_properties(dqsim.hook PROPERTIES SKIP_RETURN_CODE 77 LABELS gamedata)

  # Fidelity against pristine upstream QuakeSpasm, built by
  # tools/fetch_fidelity_reference.sh (local only; skips when absent).
  set(DQSIM_FIDELITY_REF "${CMAKE_CURRENT_SOURCE_DIR}/.fidelity/quakespasm/Quake/quakespasm"
      CACHE FILEPATH "unmodified QuakeSpasm binary for the fidelity test")
  dqsim_add_test(test_fidelity)
  add_test(NAME dqsim.fidelity
    COMMAND test_fidelity ${DQSIM_TEST_GAMEDATA} ${DQSIM_FIDELITY_REF}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/fidelity_inputs.txt)
  set_tests_properties(dqsim.fidelity PROPERTIES SKIP_RETURN_CODE 77
    LABELS gamedata TIMEOUT 360)
endif()
