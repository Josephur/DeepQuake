option(DQSIM_SANITIZE "Build our code and tests with ASan and UBSan" OFF)

add_library(dqsim SHARED
  src/dqsim.c
)
target_include_directories(dqsim PUBLIC include)
target_compile_definitions(dqsim PRIVATE
  DQSIM_BUILDING
  DQSIM_VERSION_STRING="${PROJECT_VERSION}"
)
target_compile_options(dqsim PRIVATE -Wall -Wextra -Werror)

if(BUILD_TESTING)
  add_executable(test_version tests/test_version.c)
  target_link_libraries(test_version PRIVATE dqsim)
  target_compile_options(test_version PRIVATE -Wall -Wextra -Werror)
  add_test(NAME dqsim.version COMMAND test_version ${PROJECT_VERSION})

  # Sanitizers go on the tests only: a sanitized shared library cannot be
  # loaded from an unsanitized Python interpreter.
  if(DQSIM_SANITIZE)
    target_compile_options(test_version PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(test_version PRIVATE -fsanitize=address,undefined)
  endif()
endif()
