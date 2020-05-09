#include "macos.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <Cocoa/Cocoa.h>
#include <mach-o/dyld.h>
#include <mach/mach_time.h>
#include <unistd.h>

#define UNUSED_VAR(x) ((void)(x))

typedef enum {
    FORMAT_LDR,
    FORMAT_HDR
} format_t;

typedef enum {KEY_A, KEY_D, KEY_S, KEY_W, KEY_SPACE, KEY_NUM} keycode_t;

typedef enum {BUTTON_L, BUTTON_R, BUTTON_NUM} button_t;

typedef struct {
    void (*key_callback)(window_t *window, keycode_t key, int pressed);
    void (*button_callback)(window_t *window, button_t button, int pressed);
    void (*scroll_callback)(window_t *window, float offset);
} callbacks_t;

typedef struct {
    format_t format;
    int width, height, channels;
    unsigned char *ldr_buffer;
    float *hdr_buffer;
} image_t;

struct window {
    NSWindow *handle;
    image_t *surface;
    /* common data */
    int should_close;
    char keys[KEY_NUM];
    char buttons[BUTTON_NUM];
    callbacks_t callbacks;
    void *userdata;
};

static NSAutoreleasePool *g_autoreleasepool = NULL;


@interface WindowDelegate : NSObject <NSWindowDelegate>
@end

@implementation WindowDelegate {
    window_t *_window;
}

- (instancetype)initWithWindow:(window_t *)window {
    self = [super init];
    if (self != nil) {
        _window = window;
    }
    return self;
}

- (BOOL)windowShouldClose:(NSWindow *)sender {
    UNUSED_VAR(sender);
    _window->should_close = 1;
    return NO;
}

@end

static void handle_key_event(window_t *window, int virtual_key, char pressed) {
    keycode_t key;
    switch (virtual_key) {
        case 0x00: key = KEY_A;     break;
        case 0x02: key = KEY_D;     break;
        case 0x01: key = KEY_S;     break;
        case 0x0D: key = KEY_W;     break;
        case 0x31: key = KEY_SPACE; break;
        default:   key = KEY_NUM;   break;
    }
    if (key < KEY_NUM) {
        window->keys[key] = pressed;
        if (window->callbacks.key_callback) {
            window->callbacks.key_callback(window, key, pressed);
        }
    }
}

static void handle_button_event(window_t *window, button_t button,
                                char pressed) {
    window->buttons[button] = pressed;
    if (window->callbacks.button_callback) {
        window->callbacks.button_callback(window, button, pressed);
    }
}

static void handle_scroll_event(window_t *window, float offset) {
    if (window->callbacks.scroll_callback) {
        window->callbacks.scroll_callback(window, offset);
    }
}


static void create_menubar(void) {
    NSMenu *menu_bar, *app_menu;
    NSMenuItem *app_menu_item, *quit_menu_item;
    NSString *app_name, *quit_title;

    menu_bar = [[[NSMenu alloc] init] autorelease];
    [NSApp setMainMenu:menu_bar];

    app_menu_item = [[[NSMenuItem alloc] init] autorelease];
    [menu_bar addItem:app_menu_item];

    app_menu = [[[NSMenu alloc] init] autorelease];
    [app_menu_item setSubmenu:app_menu];

    app_name = [[NSProcessInfo processInfo] processName];
    quit_title = [@"Quit " stringByAppendingString:app_name];
    quit_menu_item = [[[NSMenuItem alloc] initWithTitle:quit_title
                                                 action:@selector(terminate:)
                                          keyEquivalent:@"q"] autorelease];
    [app_menu addItem:quit_menu_item];
}

static void initialize_path(void) {
    char path[256];
    uint32_t size = 256;
    _NSGetExecutablePath(path, &size);
    *strrchr(path, '/') = '\0';
    chdir(path);
    chdir("assets");
}

static void create_application(void)
{
    if (NSApp == nil)
    {
        g_autoreleasepool = [[NSAutoreleasePool alloc] init];
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        create_menubar();
        [NSApp finishLaunching];
    }
}

@interface ContentView : NSView
@end

@implementation ContentView {
    window_t *_window;
}

- (instancetype)initWithWindow:(window_t *)window {
    self = [super init];
    if (self != nil) {
        _window = window;
    }
    return self;
}

- (BOOL)acceptsFirstResponder {
    return YES;  /* to receive key-down events */
}

- (void)drawRect:(NSRect)dirtyRect {
    image_t *surface = _window->surface;
    NSBitmapImageRep *rep = [[[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:&(surface->ldr_buffer)
                          pixelsWide:surface->width
                          pixelsHigh:surface->height
                       bitsPerSample:8
                     samplesPerPixel:3
                            hasAlpha:NO
                            isPlanar:NO
                      colorSpaceName:NSCalibratedRGBColorSpace
                         bytesPerRow:surface->width * 4
                        bitsPerPixel:32] autorelease];
    NSImage *nsimage = [[[NSImage alloc] init] autorelease];
    [nsimage addRepresentation:rep];
    [nsimage drawInRect:dirtyRect];
}

- (void)keyDown:(NSEvent *)event {
    handle_key_event(_window, [event keyCode], 1);
}

- (void)keyUp:(NSEvent *)event {
    handle_key_event(_window, [event keyCode], 0);
}

- (void)mouseDown:(NSEvent *)event {
    UNUSED_VAR(event);
    handle_button_event(_window, BUTTON_L, 1);
}

- (void)mouseUp:(NSEvent *)event {
    UNUSED_VAR(event);
    handle_button_event(_window, BUTTON_L, 0);
}

- (void)rightMouseDown:(NSEvent *)event {
    UNUSED_VAR(event);
    handle_button_event(_window, BUTTON_R, 1);
}

- (void)rightMouseUp:(NSEvent *)event {
    UNUSED_VAR(event);
    handle_button_event(_window, BUTTON_R, 0);
}

- (void)scrollWheel:(NSEvent *)event {
    float offset = (float)[event scrollingDeltaY];
    if ([event hasPreciseScrollingDeltas]) {
        offset *= 0.1f;
    }
    handle_scroll_event(_window, offset);
}

@end


static NSWindow *create_window(window_t *window, const char *title,
                               int width, int height) {
    NSRect rect;
    NSUInteger mask;
    NSWindow *handle;
    WindowDelegate *delegate;
    ContentView *view;

    rect = NSMakeRect(0, 0, width, height);
    mask = NSWindowStyleMaskTitled
           | NSWindowStyleMaskClosable
           | NSWindowStyleMaskMiniaturizable;
    handle = [[NSWindow alloc] initWithContentRect:rect
                                         styleMask:mask
                                           backing:NSBackingStoreBuffered
                                             defer:NO];
    assert(handle != nil);
    [handle setTitle:[NSString stringWithUTF8String:title]];

    /*
     * the storage semantics of NSWindow.setDelegate is @property(assign),
     * or @property(weak) with ARC, we must not autorelease the delegate
     */
    delegate = [[WindowDelegate alloc] initWithWindow:window];
    assert(delegate != nil);
    [handle setDelegate:delegate];

    view = [[[ContentView alloc] initWithWindow:window] autorelease];
    assert(view != nil);
    [handle setContentView:view];
    [handle makeFirstResponder:view];

    return handle;
}


static image_t *image_create(int width, int height, int channels, format_t format) {
    int num_elems = width * height * channels;
    image_t *image;

    assert(width > 0 && height > 0 && channels >= 1 && channels <= 4);
    assert(format == FORMAT_LDR || format == FORMAT_HDR);

    image = (image_t*)malloc(sizeof(image_t));
    image->format = format;
    image->width = width;
    image->height = height;
    image->channels = channels;
    image->ldr_buffer = NULL;
    image->hdr_buffer = NULL;

    if (format == FORMAT_LDR) {
        int size = sizeof(unsigned char) * num_elems;
        image->ldr_buffer = (unsigned char*)malloc(size);
        memset(image->ldr_buffer, 0, size);
    } else {
        int size = sizeof(float) * num_elems;
        image->hdr_buffer = (float*)malloc(size);
        memset(image->hdr_buffer, 0, size);
    }

    return image;
}

window_t *CreateWindow()
{
    printf("*** create window ***\n");
    create_application();
    initialize_path();
    window_t *window;
    window = (window_t*)malloc(sizeof(window_t));
    memset(window, 0, sizeof(window_t));
    window->handle = create_window(window, "my render", 600, 600);
    window->surface = image_create(600, 600, 4, FORMAT_LDR);
    [window->handle makeKeyAndOrderFront:nil];
    return window;
}

int WindowClosed(window_t *window) {
    return window->should_close;
}

static void present_surface(window_t *window) {
    [[window->handle contentView] setNeedsDisplay:YES];  /* invoke drawRect */
}

void ShowWindow(window_t *window, color_t *color) {
    printf("*** show window ***\n");
    image_t *dst = window->surface;
    int width = dst->width;
    int height = dst->height;
    int r, c;

    for (r = 0; r < height; r++) {
        for (c = 0; c < width; c++) {
            int flipped_r = height - 1 - r;
            int src_index = (r * width + c) * 4;
            int dst_index = (flipped_r * width + c) * 4;
            unsigned char *dst_pixel = &dst->ldr_buffer[dst_index];
            dst_pixel[0] = color->r;  /* red */
            dst_pixel[1] = color->g;  /* green */
            dst_pixel[2] = color->b;  /* blue */
        }
    }


    present_surface(window);

    while (1) {
        NSEvent *event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                            untilDate:[NSDate distantPast]
                                               inMode:NSDefaultRunLoopMode
                                              dequeue:YES];
        if (event == nil) {
            break;
        }
        [NSApp sendEvent:event];
    }
    [g_autoreleasepool drain];
    g_autoreleasepool = [[NSAutoreleasePool alloc] init];
}
