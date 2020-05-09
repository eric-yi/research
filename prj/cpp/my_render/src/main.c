#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "macos.h"

int main(int argc, char *argv[])
{
    printf("****** my render ******\n");
    window_t *window = CreateWindow();
    printf("*** window: %p\n", window);
    color_t color;
    color.r = 0;
    color.g = 255;
    color.b = 120;
    while (!WindowClosed(window))
    {
        if (color.r > 255)  color.r = 0;
        if (color.g < 0)    color.g = 255;
        if (color.b> 255)   color.r = 120;
        if (color.b < 0)    color.r = 0;
        color.r++;
        color.g--;
        if (color.b >= 120) color.b++; 
        else                color.b--; 
        printf("****** running, color: %d, %d, %d ******\n", color.r, color.g, color.b);
        ShowWindow(window, &color);
    }

    return 0;
}