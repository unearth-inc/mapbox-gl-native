#import <Foundation/Foundation.h>

#import "MGLTypes.h"

NS_ASSUME_NONNULL_BEGIN

// Event types
extern NSString *const MGLEventTypeAppUserTurnstile;
extern NSString *const MGLEventTypeMapLoad;
extern NSString *const MGLEventTypeMapTap;
extern NSString *const MGLEventTypeMapDragEnd;
extern NSString *const MGLEventTypeLocation;
extern NSString *const MGLEventTypeNavigationTurnstile;
extern NSString *const MGLEventTypeNavigationDepart;
extern NSString *const MGLEventTypeNavigationArrive;
extern NSString *const MGLEventTypeNavigationCancel;
extern NSString *const MGLEventTypeNavigationReroute;
extern NSString *const MGLEventTypeNavigationFeedback;

// Event keys
extern NSString *const MGLEventKeyLatitude;
extern NSString *const MGLEventKeyLongitude;
extern NSString *const MGLEventKeyZoomLevel;
extern NSString *const MGLEventKeyGestureID;

// Gestures
extern NSString *const MGLEventGestureSingleTap;
extern NSString *const MGLEventGestureDoubleTap;
extern NSString *const MGLEventGestureTwoFingerSingleTap;
extern NSString *const MGLEventGestureQuickZoom;
extern NSString *const MGLEventGesturePanStart;
extern NSString *const MGLEventGesturePinchStart;
extern NSString *const MGLEventGestureRotateStart;
extern NSString *const MGLEventGesturePitchStart;

typedef NS_DICTIONARY_OF(NSString *, id) MGLMapboxEventAttributes;
typedef NS_MUTABLE_DICTIONARY_OF(NSString *, id) MGLMutableMapboxEventAttributes;

@interface MGLMapboxEvents : NSObject

+ (nullable instancetype)sharedManager;

// You must call these methods from the main thread.
//
+ (void)pushEvent:(NSString *)event withAttributes:(MGLMapboxEventAttributes *)attributeDictionary;
+ (void)ensureMetricsOptoutExists;
+ (void)flush;

@end

NS_ASSUME_NONNULL_END
