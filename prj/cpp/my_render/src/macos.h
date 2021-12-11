#ifndef _MACOS_HEADER__
#define _MACOS_HEADER__

typedef struct window window_t;

typedef struct
{
    int r;
    int g;
    int b;
} color_t;

window_t *CreateWindow(void);

int WindowClosed(window_t *window);

void ShowWindow(window_t *window, color_t *color);

#endif
