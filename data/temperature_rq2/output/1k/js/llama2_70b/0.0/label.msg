Insure that tracks have a default color
fix connection pool dying, closes #<I>
[FIX] sap.m.ResponsivePopover: begin and end buttons now are with type transparent\n\nWhen a button with different type was added to the beginButton or\nendButton aggregation it remains with its default type on Phone and it was\nrendered with different color. This was inconsistent with Desktop\nbehavior.\n\nFixes <URL>
Fix clipAction loop method in hand controls.\n\n`THREE.PingPong` is not a defined constant. `THREE.LoopPingPong` is, but does not exhibit the desired behaviour. `THREE.LoopRepeat` seems to be correct.
fix: Stub constants that updated fastfile requires
fixes an issue where the test title was not printing the report name
Improved quality by encapsulating methods
ensure tests do not fail because of faulty cleanup
Strip the timestamp of url
Demonstrate failing test for missing template
fix: normalize child nodes in some cases when resolving the model nodes
Simplified check for IE browsers 8 and 9.
[windows8] commandProxy was moved
Fix: Correct check for webpack 4\n\nFix #1
User proper errors in IFrameApi
fix(require): fix directory path in require\n\nThe relative path to ui-layout.js was missing the forward slash to make it relative to the current directory. The Devils is in the details.
#<I> On the fly form validation works in Opera as well (from r<I>)\n\ngit-svn-id: svn://svn.silverstripe.com/silverstripe/open/modules/sapphire/trunk@<I> <I>b<I>ca-7a2a-<I>-9d3b-<I>d<I>a<I>a9
Removed keygen require in conf/cli.
Expose UUID in request file object.
"this" is already a jQuery object.
Fix #<I>: Job preview does not work\n\n* Correct javascript to only update description iframe.
[fix] migrate to stack-chain <I>
fix(rkPicker): change animation type to 'fade'
changes for the new api in stylecow
Issue #<I> - enter key handling for form select element, focus CSS for form buttons
Fix #<I>: Sorting in reflow mode (#<I>)\n\n* Fix #<I>: Sorting in reflow mode\n\n\n\n* Fix #<I>: Sorting in reflow mode\n\n\n\n* Fix #<I>: Sorting in reflow mode
Added id to the rendered region
Added an empty line (for the help).
[IconButton] Take advantage of the CSS inheritance (#<I>)
selection preservation off by 1, fix #<I>
Simplifies `SelectorsOptimizer` instantiation.
pass style overwrites to lines()
live.tab only change value on pointer move when actual underlying tab is changing\n\nsee ##<I>
Stop packagePath variable from being undefined
Improve type validation for notification data and options
Fix AEGIS.agent() lookup\n\nIt now considers the address field to be an identity, just like the UUID\nfield, since most agents are still looked up via their SSH host+port.
Cleaned up the code and made it send updates less frequently.
No longer print returned nil if using -e command line flag (#<I>)\n\nfixes #<I>
send non-mobx object to playbackStarted
build: tryingt of fix windows build issue
added default values for polling intervall to reduce windows cpu usage
add as utilities instead of components. resolves #2
Breaking Change - GameEngine: remove physics initialization from options. Physics engine should now be initialized from the derived GameEngine class
added support for redis database and password (#<I>)\n\n* added support for redis database and password\n\n\n\nadded db and password options to redis configuration\n\n\n\n* fixing indentation
Deep extend the options argument\n\nWithout a deep extend, trying to override just one log method will override everything.
Whoops, bundleWebpack takes its options object as the 2nd parameter
Reconnect to the same server and port.
[INTERNAL][FIX] sap.m.RangeSlider: Change event is no longer fired on touchstart\n\nBCP: <I>\nChange-Id: I<I>c6ed0e<I>dfd<I>b<I>bda
issue #<I>, custom-marker, moved CustomMarker definition line to compile as suggested
Consider Icon type prop when fetching names
Remove explicit `rootDir`, which should be the default already.\n\nSince the default is based on the presence of `package.json` or a\n`jest.config.js`, the setting of `rootDir` to `.` should be unnecessary.\n\nRef: <URL>
refactor(sdam): ensure servers emit `close` when destroyed
added ability to bypass setting a height on slideshows by wrapping singleton in .apos-no-height
Added support for Z-SRN<I>N (#<I>)\n\n* Added support for Z-SRN<I>N\n\n\n\nbased on this disscusion:\n\n<URL>
Fixed publish confirmation button text\n\nno issue\n\n- the button text would still say "Publish & send" when no members were selected\n- changed from comparing `publishOptions.publishType` directly to using the `willPublish` and `willEmail` convenience getters so the button text better matches the action being taken
[#<I>] Fix subcontainer bounds when added to the scene graph\n\n- Crazy! Setting the z position before the child has an ancestor causes the ObservableVector to set the bounds incorrectly
Added all the CLI symbol replacements
Do not send options to toString\n\nSome native toString methods crash with invalid values
fix(frames): consume bytes before heartbeat short circuit
Added "env" variable to global variable
align with the rest and use sudo for route
Support alternative redis clients, e.g. haredis
ENHANCEMENT Marking "link" and "unlink" buttons in SilverStripe\'s TinyMCE as disabled if no link is selected (see <I>efe<I> in sapphire for details)
Add 'type' property to all 'signal' messages\n\nOnly some signaling message have 'type' properties. The others rely on the key name that is in use.\n\nHaving a 'type' in all messages improves console debug logs (since I can include the type in the output).
Rename validationObject to validationModel of edit-form controller
Re-enabled all tests in SpecRunner
fix: support pointerEvents
Add support for `client['Domain.Event']` syntax for Promise-based event type checking
rephrased some test headers
Trim trailing spaces.\n\n[ci skip]
Remove needless newline from cli so Chance output can be properly piped
this is faster than just cap[1] for some reason.
Improve options handling in streamCompare\n\nRename the argument to make it clearer that this can be an options\nobject or a compare function.  This also avoids assigning to an\nargument, which is better style.
Fixed MotionGuidePlugin to not run during a wait()
Oops, forgot to remove one argument in the test.
Hotfix issue with highlight text feature when setting default color
fix(gesture): fix conflicts with Ionic Closes #<I>\n\nCloses #<I>. Closes #<I>.
fix: fix item of vnodes is undefined in getVnodes util
feat: add getConnectionsQueue in main reducer :pencil:
[INTERNAL][FIX] sap.m.WizardProgNav: Use correct parameter for current\nstep\n\n-Bug was introduced in change: <URL>
if the table-dynamic add click was not inside the tbody will append the template row to the tbody
<I>px is too small on mobile, bump to <I>px
Fix resolved when not passed by options.
SIFTSCIENCE_URL should be a const as it is never reassigned
[api] Accept an optional "before" stream which we will pipe to the gunzip.
Update karma-minified.conf.js
removing .js in require
Merge pull request #<I> from marcjansen/fix-drawtext-regression\n\nFix a drawText regression introduced with <I>a<I>e2
Check to make sure cacheWhitelist exists before trying to use it
style: Prettify\n\n(for a white guy)
Add getLanguageId method to NullLanguageMode
Cache warnings to prevent too many occurrences\n\nA warning for unknown dependencies is logged every time a moduleName is\ncreated for converting the map, meta, and paths properties of system.\nThis leads to a large number of redundant warnings. The fix is to cache\nthe warnings so that they only occur once per moduleName. Fixes #<I>
:sparkles: Add IamService to IoC
Update strategy.js\n\nAdded default scope to authenticate method.
Scratch that, reverse it.
refactor(ui-text): fix letter-spacing variables in styles.js\n\nCloses: INSTUI-<I>\n\nFixed the error made during emotion migration where all letter-spacing values were set to normal.\nAdded the condensed and expanded versions too.
revert: optionalDependencies in details script
Better handling of whitespace in statsporn commands
Fixed an issue in StatusCodeError for browsers that don’t support Error.captureStackTrace
Add a test around css rule order
Pass visit_token and visitor_token with events
index refs propagate to non-isolated components
Fix spec for phantomjs
Fix blank lastMessage for non-text messages\n\nFor non text messages (ex: media messages and group updates), the\nlastMessage field was being populated with empty string, resulting in an\nempty message preview in the conversation list. Instead, display 'Media\nmessage' or 'Updated the group', etc...\n\n// FREEBIE
Don't overwrite newOptions.method with default\n\nThis allows other methods than GET.
Modify renderer.js\n\nDeprecated as of <I>. highlight(lang, code, ...args) has been deprecated.\n\nPlease use highlight(code, options) instead.\n\n<URL>
fix(drag): file drag and drop does not work in new chrome <I>
[ReactNative] Unbreak OSS packager JS tests
Use waitFor to check for ready event
CB-<I> fixed spec<I> for wp8\n\nTest needs to check for x-wmapp0:/ instead of file:/ in case of wp
Closes #2; remove __dirname from require call to main lib.
RESTOMM-<I> #comment Added Thai language to USSD <Language/> verb\n\n(cherry picked from commit dc4b<I>c7f9acc6f<I>ef<I>fecc2baff4efdb)
collect how many values that was used to calculate the stats
refactor: speed up injection resolution
Export as a module if possible
Adding More Whitelisted Characters\n\nThese are sometimes used in their bare format without the need for translations. It would be nice if they were included by default.
fix(serverSelection): make sure to pass session to serverSelection\n\nThe executeWithSelection changes failed to pass the session to\nserver selection, which broke mongos pinning. This ensures that\nwe pass a session to serverSelection.
Replace custom ArrayBuffer converter with Buffer.from
Double default  for webrtc trackers (now Chrome <I> is released) #<I>
Switch to esm alias
gateio fetchTickers inverted base / quote volumes
Filter out fields postfixed with $* (#<I>)\n\nMongoose Map type generates paths like `fieldName.$*`. When these are included in queries, Mongo throws an exception.
Added timestamp to request #<I>
bitstamp refactor for python compatibility
debugging travis auto-deploy process... sorry for the spam
bugfix send light/aa_request only when aa found in outputs
Fixes PLAT-<I> - worker deaths in trunk
Add a project to benchmark sensing_touchingobject vs other sprites (#<I>)
fix(xod-doc): add removing tmpDir after tests and change its name to .tmp (cause its already ignored in .gitignore)
Fix for error formatting having incorrect column offsets
After calling app.quit, make sure we are not doing additional work (#<I>)
fixed issue with links on touch devices (tap wasn't working)
Split name/value only on first ':'
Prevent OverviewMap.destroy() from running twice, which was causing test failures. Change has been reviewed and approved by crschmidt.\n\ngit-svn-id: <URL>
Jetpack Connect: Prepend /connect to type and topic routes (#<I>)
Remove unused `path` dependency from createCacheKeyFunction.js
coinmarketcap workaround in test.js
Add failing test to attribute filtering\n\n@probablyup some of your changes in validAttrs.js broke the filtering\nfor invalid props. Mind taking a look at fixing this?
updated to check for commentend on same line
fix(core): improve code coverage, code quality
[minor] Fix a typo in the test
Fix typo in zookeeper.js\n\nCloses #<I>
Added errno to globals
SvnLibraryNode should inherit ScmLibraryNode
add oracles to coverage migrations
prevent line numbers from confusing click-position-detection
Use object-assign polyfill for object creation
Use outerHtml to make sure we write `<html>` element from extensions too
CKEditor: Set twig syntax {{ }} and {% %} as protected
fix: allow traversing reactive objects which are sealed (#<I>)
fix(analytics-core): added dynamic length of attr key for slicing
fix(Standalone): Ensure to use proper CLI params parser
turn arguments without a protocol into file:// urls
coss ccxt.js reverted (auto-built with `npm run export-exchanges`)
Use the block key, not the entity key.
Rearranged the attribute stack in MJMLColumnElement.js
Cannot read property 'forEach' of undefined\n\nAvoiding TypeError: Cannot read property 'forEach' of undefined\n\nUsing some cameras the getScopes core functionality can return an object like data: { GetScopesResponse: '' }.\n\nAn example is using the application Android IP Webcam: <URL>
Added repeated looking-for-peers message to close feedback loop during initial peer search period
Represent remote address in tcp server rinfo, not local
[redux] fix Reducer state type (#<I>)
Fix def.parse to actually return something when parsing object type
chore: Use tilde range dependencies for stable releases (#<I>)
Fix issue #<I>: Type Error when hover outside of world cloud container
correction for selected row (#<I>)
fix(Step): fixed typo in Step.Group className (#<I>)\n\nfixed typo of invalid comma for the generated class unstackable
fix issue with undefined prev selection
Replace wrong use of path.normalize
Fallback to PhanthomJS if CHROME_BIN is not set 🔙
Cache module relative paths to speed up builds with lots of modules & chunks.\n\nIt seems like AggressiveSplittingPlugin generates relative paths for each (module,chunk) pair, so it was taking up a lot of time.
Change format of return object of build-contract function
chore(select): removes duplicate declaration for `selectMenuCtrl`
Fixes bug with undefined console in browsers
Fixed issue with missing skiplink on async loading.
[refactor] Use preacher-api for Logs
fix(replset): correct legacy max staleness calculation\n\nWe were calculating lastUpdateTime using nanoseconds instead of\nmilliseconds, despite calculating maxStaleness with a millisecond\nlastWriteDate
Update config initializer (#<I>)
fix: fix Sidebar link active logic (#<I>)
kucoin fetchLedger minor typo
backport/fix(GuildDelete): disconnect voice and cleanup GuildChannels
Fixed a few minor errors in the test code
[xml mode] Make it actually possible to disable multilineTagIndentPastTag\n\nCloses #<I>
Cleaned up createConnection test.
fix(Messages): undeleting themselves (#<I>)\n\nwhen a _patch() is queued
added initial input to index.js
ESLint: Enable `node/no-unpublished-import` rule
bcex timeInForce
refactor: simplify expression. (#<I>)
upadated to include config module
Removed last vestiges of old "information panel".\n\nCause error message for failure to connect to mongo server to appear in abort\nbutton.\n\nCloses #<I>.
Remove old libraryUrl due to conflict with Content Upgrade
Expose constants and documentLoader.
fix: double-backslash escape for #<I>
Rename misleading function name that handles events in general, not only keydown
Fix the input field for export ontology
remove vestigial code fragment from comment \n\ncomment contains '); which seems to be a remnant of some old code, it is out of place now in the comment
scale line width by upsampling factor\n\nWithout this, lines appear thinner when rendered with manual AA.
Can't rely on outerHTML for Firefox < <I>. Use cloneNode instead.
Remove underscore dependency in "arrays".
adding grunt task for sailslinker
Make tab.setPosition not do anything if nothing is passed
Make sure `fields` prop is an array.
Fixed `zeit.world` nameservers check (#<I>)\n\nInstead of checking if the nameservers are from a known list we\n\ncan just check that every nameserver set for the domain ends with\n\n.zeit.world.
fix for handling person email google calendar IDs
fix(VirtualScroll): Don't scroll first row of a QTable when the position is on top and there is header (#<I>)
Fix roles loading (#<I>)
fix(hmr): HMR log is sometimes trimmed (#<I>)
Remove fontSize comments\n\nNot really as useful when we aren't publishing the config, plus we\ndon't list them in the docs either.
docblocks for sampler gain and pan\n\nalso removed addressed todo
No need to store trace._stackObj in LRU cache
Changes Setting.get() to accept a default value if setting is not found
test(testing): changed path to config module import
Fix infinite loop in save (#<I>)
fix(gui): disable control buttons on retry test and suite
add test for "mixing in" sistyl instances using .set
fix(build): only watch src and test directories (#<I>)
Fix missing line-breaks in stack traces lacking callee
Make Location href setting relative
Add lunr.utils.asString function.\n\nDelegates to `Object.prototype.toString` on all object except for null\nand undefined which return an empty string. This behaviour is more\ndeisreable (to me) than having `undefined` and `null` converted to the\nstrings `"undefined"` and `"null"`
Ok, opacity is strictly numeric.
add core team as copyright holder
fix issue #<I> (#<I>)\n\n* fix issue #<I>\n\n\n\n* remove deselect in service
Use new NSFW logic (#<I>)\n\n* Use NSFW value returned by discord api\n\n\n\n* Updated nsfw check
fix: fix Webpack CSS Modules issue
fix problem  in firefox with empty file upload
Update comment to more closely reflect the way we look for focusable elements.
check updatedExisting flag in mongo response
TASK: Supress warnings regarding deprecated editor identifiers in development for now
remove window from the IIFE to support node
Remove hardcoded `isDevelopingAddon`\n\nGenerally speaking, this override should not be check in...
only read textContent for aria callout #<I>
Quick fix for typescript import crash. (#<I>)
Use the provided timeout when making requests.
Adds spec.finish call at the end of before/afterEach
fixing failing tests Admin_basketAdminTest.testFrontendPersParamSaveBasket and Admin_functionalityInAdminTest.testVariantsInheritsSelectionLists\n\ndo not set form fnc value if none are passed as parameter
Use EventTarget for events latching
added check for window in Common.now
Fixed the scroll position jumping when trying to set focus in IE.
Always add transaction on connection.send if it's not present
goog.labs.Promise.LONG_STACK_TRACES: make sure that Error.prototype.stack is really a string.\n\n(Using the V8 API can turn Error.prototype.stack into an array of CallSite objects. See xrefed bug.)\n-------------\nCreated by MOE: <URL>
BUG disable archived pages from being droppable\nFixes <URL>
Add snapcraft.yaml to copyright filter
chore: add local eslint rule to forbid import from local packages
Removes the domain renderer as it's currently not used
fix: include credentials in requests and add accept json header
fix clone() in connection with relations #<I>
added more timeframes to hollaex exchange class
Add deprecation comment about callback.fired.
Update Retrieve example to create a free client
Update channel mapping ot use process.cwd()
fix(cerebral): fix returning computed from dynamic getter
Format code and remove leading indentation
Test that we do not warn when the necessary options are provided
harden queryoptions open/close test (#<I>)
Fix route/bearing param
Removing the "part1" part
Adding test to ensure nested keys for conditionals work correctly
Taking a proxy option into account
Hotfix - change var for const
Close ws with proper error code and message on stream error
Simply was calling the wrong property in my sanitizer
pass referenceWindow to percentageHook
cq is always async
split up types.ensureBlock into types.toBlock
feat(history): enable usage as plugin
Fixing typo in `find.js`
Don't open when clicking with the right mouse button
update node type for implication symbols to be symbol not binary operator
check for bugs affecting attributes, overwritten by misnamed elements references
AndroidPlatform: Crosswalk <I> is now supported, warn only for <I>+
fix: temporary disable schema validation CLI on master branch
typo in a comment\n\nJust learning Ractive by reading its source code. :)
fix: lets unify and use console
use new version of `inux` where `handle*` must be explicitly called
{Update: catena.js} Validate src and dest before starting to build file list.
The default host name should be "localhost"
[FIX] sap.m.ResponsivePopover: ariaDescribedBy added\n\nChange-Id: If<I>f0d<I>f<I>bfcb<I>b<I>d<I>d<I>
test(login): clear localStorage before logging in
html5 audio: fix seekbar not usable on playback (refs #<I>)
Add `rename` to index
remove $q from injected dependencies as it is not used
Get first tag when updating button pane status\n\nFor block elements, such as the blockquote, the status of the buttons in the toolbar is not updated, if the element is empty because is always taken the parent element.
[FIX] sap.m.MultiInput: Input width is now properly calculated\n\nChange-Id: I<I>ae<I>d<I>d<I>ba<I>d<I>b6aedf<I>cae9cf<I>b\nBCP: <I>
allow action to accept promise in 2nd or 3rd arg
Make sure old vscode-resource scheme is still corsEnabled\n\nFixes #<I>
CB-<I> fix target attribute being ignored for images in config.xml. This closes #<I>
Click awayable component should always unbind even listener when unmounting, regardless of manualBind setting
label generation: generate label on blur vs focusout\n\nThis change is being made to improve the ability to automate\na test for it.  From various manual tests, it appears to behave\nthe same as it did before the change.
Looks like a race condition in the test suite itself. Upped timeout to give time for handler to be registered before event would be fired on the low powered travisCI vms.
reset after getting the results to allow multiple app runs\n\n(fixes issue #<I>)
avoid race conditions that can lead to multiple copies of render mode GL programs
check that window.localStorage exists
more left margin for fitting axis-with-labels
add peer/seq to idxbuild
change pop and unshift to index references to preserve all of the lines instead of deleting them by using shift and pop which manipulate the array in place.
Sort emissions in `froMEmissions` helper
[touch] improve swiping experience\n\nReset the start X position when swiping in the wrong direction.\nSo that the swipe right away when the user realise that is his going\nin the wrong directino.
Replace removed background function\n\nThis function has now been removed and made and option within other\nfunctions, I've moved it within flatten.
Fixed for cases with non-lowercase mimeType
feat(dropdown): add quietMillis support
Fix a precision issue\n\nFill an input value with zeros if its decimal part length is shorter\nthan a precision specified
catch cases where raster textures don't load
fix(gitfs): avoid divide by zero
remove catch(), because addFile is sync now
fixes jshint warning: undefined variable "api"
Include jQuery when deploying\n\nThis is because liquid-fire requires it
Use the dashboard ID for the iterator key\n\nIterator keys should use something unique, like an ID :).
feat(callback): pass result to callbacks
#<I> Initial state of disabled is ignored
feat(hash): remove breaking change for bcrypt driver
fix(availity-angular): uploads callback fix
Fixes karma bug that made the tests run twice (#<I>)
Add test for laying out a short cycle
Put express in production mode by default\n\n * This can be overridden with the NODE_ENV environment variable but we\n   shouldn't rely on that being set (it wasn't in prod this whole time).\n\n * Express caches views in production.\n\nChange-Id: Ia7e<I>a5ae<I>f4e<I>c<I>a<I>feb<I>c<I>
Fix #3 Start at <I> max files, then back off on EMFILE
[map] preserve source type if it's an array
Renamed the test to 'placeholder'
added a comment explaining why we aren't using jquery deferreds for the done loading notification
Tweak EnvStressTest concurrency + repeats.
Check & warn if pools use SQLite3 :memory: databases (#<I>)\n\nThe sqlite3 driver does not support sharing an in-memory database\n\namong multiple clients, and I don't think that really makes sense\n\nanyways.
Fix(Travis): Commit before push
urlencoded: fix bug introduced in <I>e<I>a<I>
[CoreBundle] calendar.js - time format\n\nWhen adding an event with a begining time of <I>h<I> for exemple, the summary added to the calendar show [<I> event title].\n\nChanging ```timeFormat: 'H(:mm)'``` to  ```timeFormat: 'H:mm'``` seems to resolve the problem (not much documentation found in official website)
Add test for FormControl sizing
fix silent failing test (#<I>)
autofocus Password field only on toggling of visibility
Use a non-minified substance bundle in browser test-suite for sake of debuggability.
Use innerHTML in a div, not a p
Add comment to warn future users not to change the serialization to array of keys first before removing.\n\nRELNOTES: Add comment to clear function in IterableMechanism\n\n-------------\nCreated by MOE: <URL>
Add example code and tidy up other parts of file
Remove spread operator, it was a mistake
Set gitter Expires to <I> secs later\n\nFor debugging purposes for CloudFlare.
fix: title cannot be number at front matter (#<I>)
get rid of esprima require
Fix IDBIndex.keyPath type\n\nThis may return any type which was specced in [W3C Last Call Working\nDraft <I> May\n<I>](<URL>
Add factional tests to number-zero-length
Quick fix to method removeTransform() of Collection
Bugfix regarding caching and options \n\nMy previous PR introduced a bug that caused translation options to no longer be processed.\n\n\n\nAlso includes a (small) performance increase by avoiding extending the options object on every $digest if not necessary.
resource.save pass resource to cb
assert /:path does not match /\n\ncloses #<I>
Check keyPath or options after schema and fn.
html5 audio: stop buffering data if playback is stopped (fixes #<I>)
Run window.paypalCheckoutReady after setting up interface entirely
fix(processor): fixed dismantle to trigger 'attack' action log entry
Make the editing marker on mobile fit from top to bottom as it does on desktop. (#<I>)
fix test to actually use the polyfilled version
Convert internal Set/Map to arrays in IBase.toJSON()
Flush is mandatory for subclasses of BufferedRead (#<I>)
Add additional examples to passport.authenticate().
new issue have been risen with the corrected node traversing on the ui side\n\n\nFormer-commit-id: <I>f<I>c<I>aba7c<I>ed<I>abf6
Restore mocked simple.Promise after test runs
Add test for cursor losing their connecion - #<I>
test(cursor): Add missing tests
Fix Theme Showcase server-side redirects (#<I>)\n\nThe res field in the iso-router, which is used for theme routes server-side redirects, was removed in #<I>. This PR reinstates it to fix broken theme redirects.\n\n\n\nFixes #<I> \n\nFixes #<I>
fix issue with invalid reference to apiKey
allow white spaces at the end of plural forms regex
Fixed binding error on _.each iterator
email create contact in b2b mode
Get those crazy console logs out of there....
fix bug with resize image to 0px after it was loaded\n\nissue #<I>
Assign alias.length with Object.defineProperty\n\nThis is mandatory in strict mode because that property is not writable\nby default.
fix wrong function name\n\nshould be using pickBy; pick alone results in errors being ignored
support dom-ready functions with shariff-native dom library
StreetView - do not reposition if change comes from SV navigation
fix: missing new line in home demo
click tab when already selected to refresh a view (scrolls to top)\n\nfixes #<I>
fix(gomod): comment out relative replace, don’t delete
[fix] Don't crash on `/drones/running` API request
Destroy 2d contexts only if they are not the window-global canvas 2d context
Add invariant messages to socketsync
Increase entropy when creating bip<I> mnemonic
Fix 2nd occurence of right & bottom property of mock getBoundingClientRect
spec: Do not test window positions\n\nThey were too flaky.
Add functions for retrieving the view center and position
test: race condition makes test fail randomly (#<I>)
Re-export from src not dist
Fixed program.connection_found setter.
Do not clobber existing temporary files\n\ncloses #<I>
socket: added `write` alias for `send`.
finally clause is unneedded
lint is mother; lint is father
Prepend daemon log messages with timestamp\n\nThis will help when looking at the logfile at a later time.
fix: stop propagation for test indeterminate clicking (#<I>)
fix(environment): remove unnecessary/harmful internal `is_in_app` flag\n\nThis was only used to perform custom path operations that aren't reliably, and\neasily doable. Using `require.resolve` is applicable for all these use cases.\nI'm removing the flag, because it isn't used throughout the code-base now.\nIt won't invite further misuse this way.\n\n* * *\n\nThis commit was sponsored by The Hoodie Firm.\nYou can hire The Hoodie Firm:\n\n<URL>
Fix _up in detached stack props
Update up to changes in es5-ext
feat(QIcon): recognize FontAwesome Pro "Kit Uploads" icons #<I>
fix: auto create cli folder (#<I>)
Add support for showing all sections at once
refactor(navigation-instruction): use viewPortInstructions to collect child instructions instead of navigation plan
Rubicon add meta fields to bids (#<I>)\n\n* Add microadBidAdapter\n\n\n\n* Remove unnecessary encodeURIComponent from microadBidAdapter\n\n\n\n* Submit Advangelists Prebid Adapter\n\n\n\n* Submit Advangelists Prebid Adapter <I>\n\n\n\n* Correct procudtion endpoint for prebid\n\n\n\n* Added meta fields to bids in response from Rubicon exchange
Reduce duplicated actions of getMainAndFollowers()
Use `color:rgb` to indicate that a device supports RGB colors
polling: improve url timestamp for ie<I> and allow force disabling
Bug <I> - infinite loop when attempting to find/replace based on a regex
Make apikey and add option for custom Search Input integration
fix support for .json file in embark.json
Fix CommandClient cooldown exclusion checks (#<I>)\n\n* fixed issue with CommandClient user cooldown exclusion\n\n\n\n* added parentheses
fix failing [aur] service test (#<I>)
Add settings.setupFormData for deeper customization
Removes toJS() from findSourceMatches (#<I>)
[js] Ensure quit handler is invoked with no context
Update last release of Bootstrap 3
http server: fix bug in how peer ids are returned
feat: debounceCalcDelay option
not using firstElementChild since it is not supported on fragments in IE<I>
Support 3des as a session key algorithm
set default migrations dir if missing from config file
Remove saveHandler and fileClient from AbstractEditor.
Don't hide login modal; wait for page to refresh
fix error reporting when could not fetch api
binance: fetchTicker: prevClose removed
MINOR Limiting console.log() output to explicit enabling through window.debug to avoid failing IE
feat: Ensure ember-moment is installed when installing this addon
FIX: Provide default co-ordinates for new objects/pages (fixes #1)\n\nIt appears Streetview doesn't like being initialised in the middle of the sea at 0,0. This fixes the issue for now, but perhaps there's a better solution (ideally leaving Streetview uninitialised until a location is chosen where it can be enabled).
Update version to <I>-pre<I>-dev\n\nEnyo-DCO-<I>-
correcting browser vs node discrpency in module loading.
net.upload: proper usage of FormData for IE, add file param name
Check for the existence of a handler property before trying to have serverless-offline set up an event handler.
Secondary info hash announced in DHT
Shorten f character's code
Emit URL-specific connection event to support multiple endpoints per server
PROD-<I> - Added admin screens to CMS
Fix/hack fd mapping for readable/writable streams issued by mock-fs.
fix(BasicInput): Label is selected correctly in specs\n\n[#<I>]
test: add useful info to stdout
test(storage-adapter-spec): prove the storage adapter doesn't duplicate array contents when loaded/saved/loaded again
dismiss keyboard on search cancel (#<I>)
Generate a peer:disconnected event rather than peer:leave for /leave messages sent by the signaller
Fixes syntax error\n\nThis will require a new release, was missing a comma
Prevents failed connections from listing "not found" errors.
bug fix\n\npromise handler accessed value of super-scope.
changed transitionProperties to obj, for quick access
Fix React warning about unnecessary bind (with aspects CL as well)\n\nChange-Id: Iff<I>cda<I>beecbdbb0bb<I>d9a0bf<I>c<I>
Assume space cap if blank (blank caps would otherwise not render, and IE trims the contents of XML tags, which breaks the OSK spacebar).
Rename local variable to prevent confusion with function name
tests/transaction: add test for target Buffer
refactor: update Webpack build to handle minifying Icons differently from other SVG assets
Compatible with the new async convertFields
Update code coverage target to <I>%
Hide all modules in Webpack output by default
Bug <I> - 'Generate Link' in compare editor diff page is useless\n\nHide the 'Generate Link' button in the compare editor if Orion is\nrunning on Electron. On Electron the link will always be pointing to\nlocalhost so it cannot be shared with other users.\n\nChange-Id: Iec<I>eb2d<I>f<I>dc<I>aa8c3b8c<I>df
Fix: Don't destroy ad manager on detach (#<I>)
Check before using array in READY handler (#<I>)
bump ticks left to fix Edge/IE bug
Fixes test not working on Windows.
Removed jQuery CDN info. [ci-skip]\nNow that the CDN option is in soil, there's no need for this information to be in the file (and it was outdated).
Bug <I> - Get workspace name not from req.user.workspaces, but from metastore intead ; workspaceInfo not returned from fulfill
refactor: Use imports in package index
Fixed Resizing TextField Size on Mount\n\nCloses #<I>
Hotfix for setState warning.
hls: avoid timeupdate for wrong duration/playback type (refs #<I>)
Missed article record lables.
Added tap and xunit reporters
Revert change in test as we are still returning an array
Fix off-by-one error in counting the number of messages\n\nDon't count the @metadata block in the number of messages.
ancestorPerformanceRestrictions should be chainable
add rxMode property to channel api (#<I>)
feat(shape): Implements get and set of names from shape\n\nThis feature allows to you make JSON represantion of each shape and know what is the shape in JSON
docs(airbnb example): googleMaps instantiation was somehow missing\n\nnow good
Show the actual traceback for exceptions while preparing CLI uploads.
skip the label edit with enter spec
fix chrome auto add TBODY element which may not in html
Less code, smarter function, deal with text node\n\nPreviously normalizeSpace assumed a parent node as\nargument. Now it works also when you pass a text node\n(this is the case of the paste).
gateio fetchTickers fix\n\nfor now response is array, no need to select 0 element\n\n<URL>
BUGFIX: Weird issue with backported has() function
trap for missing token object in SocketService
Add option to use "Secure" attribute on cookies\n\nThis addresses #<I> where some (most?) browsers default to `SameSite: None` and don\'t like cookies without the `Secure` attribute set
Fix typo (#<I>)
Better version range for dependency\n\nMakes us include alpha primarily
Accounting for both null and undefined properties of the objects to be sorted.
Addded polyline distance to the tooltip of a polygon.
Fix: delete unnecessary duplicated question (fixes #<I>) (#<I>)\n\n* delete unnecessary duplicated question\n\n\n\n* Fix: delete unnecessary duplicated question (fixes #<I>)
bugfix – set feature image id
feat(Textfield): Render `children` as last element to ease customization of the Textfield by client code. (#<I>)\n\nSEE Make Textfield render children #<I>
Added source as a task property\n\nUn-compiled source is now stored for transmission to a cluster or\nnetwork.
⚠️ skip file watch test on CI
Fixes broken save on other entities using tupane layout.
chinese: override method monthsInYear\n\nFixes #6
Properly select asset card when reloading page.\n\nThe asset card highlighting behavior was broken at some point;\nif you reloaded the page while playing an asset from the home page,\nit would play that asset correctly but wouldn't highlight the card.\nThis change ensures the card is selected.\n\nChange-Id: I5b<I>cc7fdb<I>f<I>f<I>b<I>d<I>e5b7cb<I>
pass relevant element into focus/blur handlers
Refactoring: Reduce cyclomatic complexity.
fix(actions): log in prevention of extension
fallback value for default position returned by a function
FIX: keys params of object property type
set utf-8 charset
feat: pass a node reference to dna core BaseComponent constructor
add noop transform method to IgnoreFormatter
Avoid accessing props in constructor
refactor(Multicopy test): No need for callback in after function
Add an additional clarifying comment to goog.global in ES modules\n\nRELNOTES: Add an additional clarifying comment to goog.global in ES modules\n\n-------------\nCreated by MOE: <URL>
<I>\n\n- fix Numeric field does not render decimal mark on input
Fix "TypeError: fetch is not a function" in Node.js environment (#<I>)
refactor(InsertBinder): render subviews before inserting element into the DOM\n\nRendering view detached from the DOM is necessary for animations that need to determine height or other geometric property of the rendered element.
Duplicate the name validation for now
Enabled ropsten test cases for Etherscan (it is back up).
Removed the lines instead of commenting them out
Focus on the first input too, in case it's not a text field
Fix canaryVersion in "shopping-list-advanced"
change function syntax\n\nI’m really nitpicking here, but since we’re assigning different\nfunctions to wrap_method() depending on browser support for foreign\nobjects, it seems more logically coherent to declare a variable and\nassign a function as its value, rather than declaring a function\ndirectly using the function keyword. So, a small style change for\nconsistency, sort of. (Consider this a companion to the previous\ncommit, the wires got crossed here and the content/messages are\nreversed for these two changes.)
[js] Fix typo in execute_script_test.js (#<I>)
[fix] Make a copy of event listeners\n\njoyent/node@<I>dc<I>fbf<I>e2e<I>e6f3baacdd5ff<I>e8de3f7 fixed a regression\nwhich caused `removeAllListeners` not to modify listeners array.
Fixed dot unstuffing\n\nRead from this in the RFC:\n\nWhen a line of mail text is received by the SMTP server, it checks\nthe line.  If the line is composed of a single period, it is\ntreated as the end of mail indicator.  If the first character is a\nperiod and there are other characters on the line, the first\ncharacter is deleted.
engineScript: use external response callback instead of the one proxied inside sandbox
Fixed spelling of "separated" (#<I>)\n\nI know it\'s super trivial, but boy my <I>th grade teacher sure did permanently burn the spelling of that word into my head.
Give the routes the capability to define custom route specific middlewares.
Fixes microsoft/monaco-editor#<I>: Fix running the build on Windows (files with \\r\\n)
Added RenderTexture.clear - because, well, it's really useful :)
chore(maintain): update grunt tasks
Closed #<I> : Now, a more meaningful error message is displayed
Fix the bug that the writing mode specified on the body element did not determine the principal (root) writing mode
Add forgotten jumpToAnchor funcitonality\n\nWas accidently removed when refactoring callOnContentReady
include default port in default server_name
Outputting a more enlightening error message from WebSocketClient when a handshake fails because a non-<I> response was received.
Do not add print commands in the `toCode` method of the `Markdown` expression
Move key definition into pmore.trigger
Add period on line <I>\n\nUpdate line <I> to be uniform with other comments
Support "dictionary" objects (created with Object.create(null)) in field-level \'$and\' and \'$or\'
Don't fake snippet mode when fetching external socket connector if\nrunning mode is proxy. re: #<I>
BUGFIX: Pagination back button always defaulted to zero
_1btcxe withdraw() signature unified
fix(shape): Fixes issue with empty arguments in constructor
Alias'ed navigation react native to web package\n\nSame idea as react native aliased to react native web. Need to npm install navigation-react-native-web when it's released - for now, manually copying the built package
Make webpack (way) less verbose
fix(draw): apply default renderer with minimum priority\n\nThis makes sure renderers do not need to provide a priority.\n\nBREAKING CHANGE:\n\n* Renderers must now be initialized _AND_ must inherit from diagram-\njs/lib/draw/BaseRenderer and invoke BaseRenderer(eventBus) in their\nconstructors.
Added a false test, just because we can :D
Fix typo (#<I>)
Clicking outside of dropdown removes select2-container-active class.\n\nCloses #<I>.
Add forceNodesVisible check to totalNodes counting
remove component checker in vdom
removed browser feature 'high bandwidth' from VideoPlayer
load closure polyfill before the Selenium objects loads the closure library
Filtering data based on type fields before sending it for saving.
inverting control of directory lookup - moving into kue-ui-client
Refactor async to promise for Node < 8 support
tests(samples): the acceptable latency for the batch settings test was a bit too low (#<I>)
Issue #<I>: Fixing problem in last, missing typeName in hook args. The correct hook interface is query(typeName, query, request, callback).
Bug <I> - Fix links to graphs in perfherder compare
first pass at smoothing on scale using gl_FragCoord.w for low-end devices
propagate rename .done in .exec in view
Fixes #<I>. Clean up inArray.
fix(ToastNotificationList): do not override class toast-notifications-list-pf (#<I>)
fix(demo): erroneous FairPlay keysystem in demo (#<I>)
Reducer tests: typo, probably a copy/paste artifact
test: Use Chrome <I> (was <I>) for tests, for Object.entries support
[ut]Ensure not to call the done() method multi times.
fix: delete duplicate content (#<I>)
Naming tweak: progressiveStep => progressive.
Activate launcheditor with direct source
Add a list of subscribers to method_missing stubs
fixed db permissions for tests to work locally and with travis
Don't use NoErrorsPlugin with React Hot Loader
Keep track of all symbols\n\nThis avoids that we add parent namespaces of symbols that were already\nadded because they have their own definition in the externs.
Added error handling for cancelled consumers (#<I>)
feat: update webpack to process png with file-loader
Added 4 funnies. (#<I>)
Fix default render option\n\nThis should make it compatible with newer React versions again.\n\nCloses #<I>
Generate UUID, store it, and attach to context
why is travis succeeding but local not...?
Changed DropButton to fix an issue with caller controlling state (#<I>)
Update JIAWEN to light_onoff_brightness_colortemp_colorxy converters
Fix typo in tests\n\n[ci skip]
Allow for the simple merging of flows for reporting
[docs] Fixed es6sample.js thresholds
Add a title in modal example
Added failing test for string ids
zip: zero is a valid compression level.
Fix builder instance event name\n\nReplace listener\'s event name from "deleteComponent" to  "removeComponent"
avoid adding spaces around ObjectExpression. closes #<I>
Don't auto-start the first scene.
test command should always exit on failed tests
Update request.js to use mailgun API v3\n\nThe new API seems more stable with better performance.\n\n\n\nSee <URL>
Enable useCapture on touchend events for iOS Cocos2d support
PWA-<I> moved translation to theme
Resolved Error: Digest method not supported
Better console log for event_target callback exc.
Bugfix: Only ask if relationship should be deleted if it actually exists
add headless runners to customLaunchers
Adding test simulating Model.update with attribute beginning matching for required properties
remove domain name and root slash from extracted id
Update test to be better approximation of real situation.\n\n* Uses mocked socket that is actually "connected".\n* Avoids InvalidStateError that was thrown by TestSocket because, though the pusher state machine was hacked, it was not actually connected.
Replace new Buffer(...) with Buffer.from(...)
fix(modal): only call toggleClass when needed (#<I>)
fix(ssr): remove trailing hash in webpack module identifier when\ngenerating client manifest
Fix deprecated computed import
Remove .only (they have no effect)
Minor: Restore default parent name AbstractNode for items which show no inheritance
renamed path to required library.
Better fail/done arguments detection for the always callback.
fixed paths in distribution build AGPUSH-<I>
FIX: bitz.fetch_ohlcv throws in python if result is none
Fixing bug with annotations being kept in memory
Exclude client assets from nodemon target
:bug: Fix for dirs with ext in name
[#<I>] - Fix pointer coordinates with DevicePixelRatio scaling
Slice arguments for computed only once. Fixes #<I>
Fixed  #<I> - Style props doesnt update after re-rendering with Inputmask
Use cryto-cacerts to use all /etc/ssl/certs PEMs
Resolve the e.g.:"C:\\"  issue on windows
Correct the attribution with the right link, icon and icon title
[Web UI] fix folder plugin decoration issue for exclamation/question mark
Test another case for the Siren integration.
Bug <I> - editor rulers act funny on html files
SONAR-<I> drop issues filter by last week
fix(server): Resolve issue with unset paths object\n\nResolve the issue brought to light after merging #<I>, whereby the\npaths.public key is not set for the server if you are specifying a\nDB url.
Fix API of 'keys' method in memory adapter.\n\nThis fix courtesy of Jake Dempsey:\n<URL>
refactor(wire-protocol): ensure stmtId is incremented on getMore
[feat] HTTP: Response class, feat. a json() method
Properly tests args for script
Bugfix: shard.linkMaster throws an error in some circumstances.
return data like before on db.exeRead
Change '일분' to '1분'\n\n- Usually korean write 1 minute as 1분 not 일분
Use slug for user mentions page
show new and hide old immediately after giving each exploded piece a\nchance to animate
Fix for: Function.prototype.apply() in IE8 chokes if second param is undefined
Fix use of inbox rather than feed
Now the number <I> is correctly completed
Added additional tests for the 'amDateFormat' filter
fix(native): fix the animated implementation inside a Modal
Fix #<I> - Not all plugins are unregistered on destroy.
[BUGFIX beta] fix #<I> with a guard
style(generator): small style change
style(js): use recommended array instantiation
NCL-<I> Add default PATCH method to all resources
release script will now update latest version with RCs
refactor(calltaker): Start a call on clicking Plan button.\n\nThis is except when the field trip window is shown.
added isPackStation as well as parsing and composing of PackStation address
update ignore so that `sudo npm link` will work
Prevent adding all layers to the map when embed is set to true
fix: undefined:<I> is whitelisted in CSP, if `options.liveReloadHost` is not set
try/catch around the mapper to fix the test
Substitions with an ImportValue was writing [object object] to vtl
feat: Update Korean shortMonthDay
Component: get rid of "__isInited" flag
document style request will add lost xmlns=xxx\n\nsubelement doesn\'t has a ns: prefix,\nso the top element must have a xmlns=xxx in addition to the existing xmlns:xxx="url"
Plugins: Display Drake select plan\n\nIn case when the user has no sites with plugins display Drake and ask\nthe user to update a site to a plan.
fix(cockpit): variable scope with TransitionInst\n\n- set correct variable scope for variables in Transition Instance\n\nrelated to #CAM-<I>
Remove e/i/s tags in live preview to emulate the PHP renderer's algorithm
viewer.js: fix typos in variable declarations
Created gpf.interface (based on gpf.define)
fix(dist): revert AoT changes. Too many issues\n\nCloses #<I>
[F] Correct karma config
Minor change to another doc comment.
Make dynamic-data example work when hosted
don't start drag if <button> is clicked
Add a delay after sending the scout.otaboot command\n\nThis allows the bootloader to start up after the reboot.
Revert a change in tag pages\n\nWould cause a bug with tag page listing
Update hibernate version to <I>.Final
chore(eslintrc): throw errors on prettier formatting rules
Unpack manners executable after installing manners-bin
(Bugfix Android) added onCancelled callback to child event listeners
refactor(spaces): fix logic in useSpaces
Improve tests for fs.readdir on github
guard against non-existent paths in <NavList>
remove the start API from the sync server
Routing to inspections list controller via hash navigation
fix(build): asset cache header now set to 1 year (#<I>)
Do not require 'body' on reflow
Add error logging to prepublish script
fix issue#<I>: scrollParent utility may enter infinite loop
Remove superfluous capture group in regexp.
Rename 'doubleClick' to 'ignoreEvent' now the method is being used to stop events in more than one situation.\n\ngit-svn-id: <URL>
add new test to ensure that #<I> stays fixed\n\nthis will test the condition on which issue #<I> was made
Updated parent method so passing a CSS selector to its parameter filters down the parents
Add a get global distribution type function to fix @CartoDB/deep-insights.js#<I>
Print an error message when new is not used\n\nInstead of instantiating a new instance when called as a function, print\nan error message to let the user know to use 'new'
simplify getParent (#<I>)\n\n`x >> 1` is the fastest way of doing `Math.floor(x / 2)` which is what\n\nwe need here
bittrex parseTrade cost and fee
Stop normalizing \\r\\n new line notation in stringToUtf8ByteArray.\n\nThis is in response to external issue\n<URL>
Use the currect coords to test for pointerWasMoved
share sass-cache directory between instances
broadcast socket id as part of slidedata
Fix to inject data using data() instead of just attr()
add buildStr to eval ruby without creating file
Avoid `thisArg` use of `_.map` for lodash v4 support.
Don't show the reply button if the user can't reply
[CoreBundle] delete line in config.js
Changed; trigger open in XHR transports right after starting polling.
coinmarketcap safeTimestamp
Adding deprecated tag to onBlockerClicked
expose a method to Datepicker, so selected date can be changed
fix chaining + allow subscribeOnce to be unsubscribed via the context
don't exit out on missing detection rules, just continue
don’t keep the process alive only for the timeout (dedicated-process strategy)
Minor bugfix: GCLI was getting the completion wrong for 'window.' because it mis-calculated the prefix
adding prorogram as parent property on options and commands
browser/index.js: Changed semicolon to comma in var list (DOC_TYPE_START and DOC_TYPE_END were globals by accident).
Replace arrow functions with regular functions\n\nThis enables support for node version <I>.x and higher.
update AJAX.upload on form
Compressor - fix compression when there are dropped entries. Fixes #6
fix issue with breakpoints being set on incorrect line when --break is passed; breakpoints now work as expected
Node <I> support (it doesn't have Number.isSafeInteger())
refactor(PageItem): Rewrite createButton return to functional component (#<I>)
fix: Rm safariViewController for Android\n\nIt works great until the call of the hide() method
Remove support for ancient `loose` field
try a longer timeout for travis-ci
added listener for curves_subset change
add parentId and type in the add method
improve tests for set (#<I>)
tooltip/popover ko bindings can be used by plugins
Fix edge case where we transform too much on deps
Test: Do not enable PostCSS by default - #<I>.
bootstrap root element initialization, changed root element to canvas for web.canvas
Update rule: `avoid-sha3`
change onclick event handler\n\nIf an user click a node, the resource uri will insert into input.
dummy last unit for AA estimation in light
Fix storage for select value\n\nBefore if an option value had an apostrophe it could do a javascript error due to the fact\nthat the value was directly used in the selector.
Fix bug with ajax update, fix bug with opn on focus and this example in html file
Pass props down to Modal.
fixed issue where 0 image ids does not reset the angular directive.\ncloses #<I>
fix missing param in createQuery (xrefsearch)
make clear what is done here in the code: the symbol name is resolved
Added msg to disconnected event
fix: Read "accept-encoding" directly from headers object\nThis is nodejs native, and enables use in non-express environments
Use utility to get a value's native class
fix CLI usage in README (#<I>)
refactor: add guard to value length closes #<I>
(#<I>) - Remove console.log from test
Fixed issue with path.resolve
Make users property in server to be a proxy. (#<I>)
Fix publicPath for assets in dev mode
fixed a small lint issue (mixed space and tabs)
Prevent infinite loop (#<I>)
fix the custom module to work when compiled
Add Inconsolata to front of code font stack
Use "style" property in package.json\n\nThis seems to be a prevailing standard already (<URL>) so use that instead of <URL>
Fix async example allowing duplicate selections (#<I>)
access: put var delclarations on their own lines
chore(CellCreatorButtons): Add Flow types to CellCreatorButtons
Use `Boolean` instead of `!!` in Flash check
Allow for 2-column display when only 2 charts (#<I>)\n\nPreviously, it would allocate 3 or more columns (space allowing), even if there were only two charts to show.
fixed bug where wrong timeouts were being wrongly timed out
Test suite passed; !cmd in PRIVMSG now allowed
fix(QBtn): disable not working correctly with type="a" + href #<I>
Account: Add a new language: Saraiki (#<I>)\n\n* Add a new language: Saraiki\n\n\n\n* Update language name
Log global and daemon error (#<I>)
Update check-build.js\n\ncross plattform compatibility
[FIX] Field: forward aria label to extrenal context control\n\nonly if this control supports an aria label\n\nChange-Id: I6e<I>c<I>b<I>a<I>f<I>c5f<I>a1b0a\nBCP: <I>
Buttons only have 4 out of 5
Add test for ignoring meta-key link clicks
Use the old values for resource names\n\nThe logic here is that anything below v3 will use the old resources, after <I>, the resources will carry a version number in their name.
config needs to be updated by the parameters passed before they can be used in the SQS constructor\n\tmodified:   lib/client.js
jquery.csv.js\n- added a quick documentation fix\n- added a newline at the end of the file
Adding tooltip to JSLint start on start-up with JSLint disabled
Document requires path (#<I>)
Parse the `package.description` as markdown #<I>
manually remove signal listeners (was breaking coverage to remove all)
Hook into afterDocumentChange.
fixed  rewrite in swig filter
adding a filter for 'utf8' for node-mysql <I>
disable console log for now until we fix the correct path for BT 4
modified return of args in promise
Initialize web3 provider before deployAll\n\nThis makes possible to load accounts before deploying the contracts.
Allow usage of both skjar1 and skjareinn, since it is referred to as skjareinn these days
refactor: from=httpsPort
Changed one instance of the depreciated ".className" to ".type" that were being used on objects returned from the getTokenAt() method that was missed during a merge.
chore(test): add test for max safe integers in ranges (#<I>)\n\nCloses #<I>
don't call the callback twice in case it fails\n\nConflicts:\n\n\tlib/mbtiles.js
Add empty default impl for onEnterKeyPress\n\nThis function is called unconditionally in handleInputKeyDown.
Navigate back and refresh the list on ignore
added method to get translation for given locale
Fix typo in test (#<I>)\n\n* Fix typo in test\n\n\n\n* <I>% Coverage
reverting. getValueAtTime should return unconverted units\n\nMaking a note of this in the docs
Implemented missing mongodb method
pass events for onFocus, onBlur, and onChange
Clarified usecases of 'done' and 'cancelled' job events\n\nno issue\n\n- Added docs around which events are suited best for job termination/completeion
TAO Items:\n * update preview to simulate the real item delivery\n * add a console in the preview to see all the interactions of the item with TAO\n\ngit-svn-id: <URL>
Update phantomjs package name.
Example how to use fn === true
dist changes for extra comma
Corrected check for opening the dropdown\n\nThis corrects the check that was used when the dropdown was opened, so\nthe results would be immediately rendered instead of delaying the\nloading. Previously it would delay the immediate load but immediately\nload the results when the search term was emptied.\n\nThis closes <URL>
Fix spec: should generate default form def from a schema\n\nFieldset defaults will include the key attributed (commit <I>ca<I>)
Increase slab limit for memcached
Ensure that watch mode always implies development mode too
<I> - Adds event trigger and bind to close action bar when sliding tree items are clicked.
fix: pass item index to "from" function prop in useTransition
bitz parseTransactionsByType console logging leftover removed + cleanup
https for Stamen (#<I> without the merge commit)
Duplicates should be tied to output paths not urls
Test Push 'n' Pull replication
[SnackbarContent] Change backgroundColor approach (#<I>)\n\n* [SnackbarContent] Change backgroundColor approach\n\n\n\n* Fix color
Adds placeholder text color as a prop
fix(tabs): fixes issue with tab pagination in Angular <I>
added cool logs to nxs
[CLEANUP beta] Only register Ember.ContainerView when legacy view support enabled
DanielWagnerHall: Import logging into session - probably not the way this should be done, but it fixes the build\n\nr<I>
Remove catchpaste from grunt
implemented sync.isForced to detect if there are force flags on a node or it's parent
Fix: endpoint for waitlist unban missing slash
Fix hot reload in IE/Edge strict mode (#<I>)\n\nFixes #<I>\n\n\n\nInspired by <URL>
test: put with `_deleted: true`
Fixes #<I>:  set table components for filter page.\n\nIn order to display the correct number of rows and selections we\nneed to set the table components manually.\n\n<URL>
docs: switch notes to use enclosed circle icons
fix(cockpit): change priority for decisions navigation tab\n\nRelated to CAM-<I>
Removed server 'error' event
Start id with underscore (hint it's private)
Force loading of UV by resetting flag in turbolinks:load event handler
refactor: refactor the export in index.js
Fix bug #<I> (Exception thrown on save if file not visible in project\ntree) - move initialization of wasOpen after the check that early-exists\nwhen $directoryNode is null.
Offset: allow small differences in offset.top\n\nFixes gh-<I>
fix(Header): Disable event listener on input fields (#<I>)
Work around bug where safari won't adjust event.key for shift\n\nIssue ProseMirror/prosemirror#<I>
removing duplicate user.inList since we call searchUsers fn in getusers
Fixes #1\n\nThere was an old "require" call that was forgotten there.
RELNOTES: Use an integer uniqueId.\n\nAny results from Math.random() below <I>+"" will be rendered with an exponent, eg <I>e-7. When this is parsed the exponent will be part of the test name, failing the test on a navigation error incorrectly.\n\n-------------\nCreated by MOE: <URL>
Fixed invalid module resolve paths in webpack
Moved to after the fetches
Don't use the closure oldLink to determine if history needs adding.\nThe current link must be checked to avoid adding to history states with the same url.\nThe closure oldLink must still be used for the cache because that was send to the server for the navigation data diff
docs: move input in storybook to forms
Update theme-manager.js\n\nAdd function to modify the zIndex of the raw theme
Added global flag to path replace function in handleRequest to allow more than one space (aka %<I>) in the name
build(browser): Add back debug func to CDN bundle (#<I>)\n\nTemporarily stop removing debug functionality from the CDN bundle while\n\nwe work on a more fleshed out bundling solution.\n\n\n\nWe previously added static globals in\n\n<URL>
Revert one logging change to remove the timestamp.
Take into account Windows paths too\n\nThe regex only matched "/index.jsx" in Unix paths ("path/to/index.jsx"), now in Windows paths too ("C:\\path\\to\\index.jsx").
Remove the testing on chatroom.queryAll method, since this API is no longer provided.
Fix missing "with" word in `it` title
Fix bug where instrumented version of webidl2 was loaded despite JSCOV not being defined.
PERF: hide discussion list pane on mobile
added significantPrecision flag to bitfinex
Do not catch errors inside user callback.\n\nWhen parser is used asynchronously, do not capture the error\nwithin the try catch block. Also, do not run an error thrown by\nthe parser through tracer smartError() unless it is a\nPegSyntaxError.
refactor(es6) upgrade MultiWatching to ES6 class
Use x-mine custom protocol handler for RubyMine
Spacing issues are annoying, fix test for crossbrowser envs.
Temp dir issues on Windows\n\nNode likes to do things for us!
More detailed logging for cinemeta call
Fix reconfiguring plugins with non-objects as options
Better exports for esm and typescript loaders
feat(yup): unit test for inclusivity
RG-<I> increase hermione subprocess STDOUT buffer size to fix tests stops
Bug <I> - Git plugin "orion.edit.diff" service should use Git.DiffLocation from the file meta data.
Add `dashPattern` option to support dashed crosshair line
fix(QPagination): [v1] QPagination - Error while entering page number #<I>
Sorted was not being passed if instantiating via function
#<I> fix broken test "should not call cancel ur" that contradicts itself
Test Refactor: public-exports.js (#<I>)
Trying to figure out why build fails in travis but not locally
Silence warnings on createPath and createHref
Fixed recursion error\n\nFunction name had been changed and can no longer be called with\nshort.generate here.  Fixed it.
Update downloadBitbucket.js (#<I>)
feat(constructor and base url): Adds a flag to the constructor that allows for use of custom domains\n\nre #6
Fix setup for HapiJS servers.\n\nSupport for most connection states
special-case ordering of min and max attrs to avoid truncating values
allow setting multiple headers in the browser as well
Update address to new public resolver contract
cover fallback to default migrations dir with a test
Set color to old color, when mouse leaves the color picker too fast #<I>
disable gpu with xvfb (#<I>)
[fix] properly working canihaz on travis
astar: optional timeout for findRoute() - you don't want to use astar without timeout in browsers
dealing with export structures for easier passing along of Twig object
Optimizing is.equal to return early if value and other are strictly equal.
fix issue with kibana config when using https
Make protectedRoutes work with paths that start with /
Just testing that merge doesn’t need to return, it augments the object
Added unique arg identifier for getCdps schema
Cleanup: Cleaned starting tst object
bug: bug(directive): fix guard checking at unbind\n\ncloses #<I>
Run prettier on test/processImage.js
Wait for startFlows to resolve before returning from loadFlow call\n\nNeeded when running with <I> or later.
Consider only the portion of a protocol prior to the + when autoloading
Replace MXFocusTrap with RestrictFocusToChildren component
Expose debug in main Tinkerhub instance
Fix Orvibo AM<I> power source (#<I>)\n\nThe battery is showing the correct value but is not recognized as a battery device.
Fixed missing 'var self = this;'
graph.attrs should be an object, not a function
Allow defining more packages in the jam section.\n\nThis helps porting over existing projects so that you don't have to\ncreate a jam package for all the dependencies that are already in the\nsource of the current project.
exit with a fatal error if a log file is passed in that doesn't exist.
fix callback typo\n\npointed out by <URL>
feat: can set the api version via the configuration
remove compiled prefix from stored contract code
wait for document to load before triggering pdf layout
Refactored variable name to match intent\n\nno issue
Fix for LazyFiltersResult: differenciate promise and value\n\nCommitting this on behalf of @dgaubert\n\nSee <URL>
Add deltaX/deltaY to the events dispatched by MouseSelector\n\nR=<EMAIL>\n\nReview URL: <URL>
Adding "shouldUpdate" to propTypes
removing Deserializer set - unused now
feat(blueprint-service): generate spec file based on the configuration file
Added another explicit setting of this.paused = false, this time in abort. Helps fix part of an issue when trying to play a single test from the executing pane (instead all of the tests in the executing queue get run)
chore: use relative path for test import
Updating ARM code from arm to armv7l\n\nElectron no longer publishes binary releases under the arm tag, and requires use of the armv7l tag instead.\n\n\n\nAs other parts of the build system (and downstream tools) may still be relying on using the "arm" keyword, I\'ve made it so that "armv7l" is passed to gulp-atom-electron without changing the internal code structure.
refactor: do maxIndex calculation without reordering
toughen test (#<I>)
Send escape key cancel events through action handler\n\nInstead of just closing straight away\n\nBug: T<I>\nChange-Id: Ieaf<I>fe<I>b<I>e0ee<I>d<I>f8f2e<I>f<I>
Fix for #<I>: playbackRate method and property\n\n.tubeplayer('playbackRate')\n.tubeplayer('playbackRate', 2)\n.tubeplayer('data').availablePlaybackRates
[index] Add component in the tree
Add missing prop in the modal docs
Add Esri Mosaic Layers, too.
Fixes the reset test after we updated the fake mac addr
remove one of the two new Function() from selector
Makes changing passwords require the user's old password
Add removeFile() to Node statics
PWA-<I> Corrected styling of filter items
allow passing optional pre-build  map path to create game
Fixed jshint complaints.
stripping tags from JSON response because of some plugins/malware that injects HTML into a response, screwing up editing\n\n\nFormer-commit-id: <I>bb1bc<I>e<I>c<I>b<I>eb<I>cf<I>
improved IE id/name conflict detection to cover more cases where GEBID is not usable, without using 'innerHTML' it is hard or impossible to set 'name' and 'type' attributes on controls in IE and expect meaningful results; even if this is just an inference for IE it seems to cover similar bugs also on Opera
ensuring annotationsFile exists before reading
Update to support React <I>
fixed the test error caused by the external CUI dependency
Update angular-timeago.js
[tests] Updated for 8c6be8d
js: templates: comment-loader: Correct typo\n\nThis is a remnant of string interpolation in pug templates\nthat was overlooked.
added test for server.raise function
[drone] use newly added .stop() callback
model => Model in resource controllers
Improve warning if the template isn\'t found\n\nPreviously we\'d only log "Invalid template (simple)", which wasn\'t\nvery helpful. Now we log "Invalid templat (simple), could not find file\n/User/somename/someproject/templates/simple.jade" which is a lot more\ndescriptive and easier to debug.
Log only the first element of content in order to\npreserve tags (<URL>).
Call back only once even with multiple workers
protocol designer: fix wellKeyToXYList, it was flipped
Don't add null changes to history\n\nIssue #<I>
Rejected promises now returns the full res
FadeImage node didn't have output pin\n\nI replaced the pin from input to output. worked fine for me.
Add not duplicated paths test for css utils
- exposing original errors to keep the original messages
If type is set, trigger updates on change of name and dbName fields
Added a special option for the new toggle closed ui.
Log if same-page button is visible after render
comment out semicolon removal for right braces in code generator - fixes #<I>
configure(): rename exposeWithPrefix setting
Fix double type conversion in bloom.multiCheck
Add support for fs.realpath.native (#<I>)
sometimes i like to commit.  I like to commit a lot of times.
Add very basic client error handling for large bins\n\nThis basically hijacks the tips notification and displays a message\ninforming the user the bin is too big.
WIP(components): ensure suggestions list is closed appropriately.\n\naffects: webiny-app-ui
Update inline docs.\n\n On branch master\n\tmodified:   lib/enginemill/environment.js
possible fix with npm run dev command /admin-lte/dist/img/boxed-bg.jpg issue
fix(@pob/root): only install ci workflows if ci is enabled
remove forceFocus: true so the page will not scroll to the middle
Starting with parseCookies method. Still not working.
Prevent gh-spin-button from infinite spin\n\ncloses #<I>\n\n- Always set showSpinnerTimeout back to null after 1 second.
better test in case we are running in a headless environment
forgot to add a docblock
Compensate for the lack of resize restriction.
Fix forgotten merge note in #<I>
Changed keyboard shortcut to ctrl+shift+alt+a
Add missing lodash import to QueryEditor
fix: #<I> (#<I>)
Adapted original posix regex to get two dots
Don't fetch on window/tab is unfocused
tinkering with right.js
Fix ref to THREE in UpdatePlayerPhysics
passing right header to card when cards is passed as json data
Fix sample for youtube search (#<I>)
Adds test to ensure #<I> is tested (#<I>)
Empty processor for packet <I> added, fix #<I>
fix: remove L.DomEvent.fakeStop from leaflet patch as it is not available nor needed anymore
contenteditable bindings should listen for blur events
aofex-fetchBalance-timestamp
mongo : return same information for stream and normal file read
Fix bufferSize to proper <I>MB default size. (#<I>)
Starting to remove some unnecessary conditions when indexing
Fixed a bug. Ensure that functions that has no name will return as anonymous funcitons
Fixes SSR errors by checking for window. (#<I>)
Add a fallback for viewport size\n\nShould fix issue #3 - getting the bounds fails when the map container is smaller than intended for the activearea div.
Import directives and struct definitions don't really count as executed code
Make sure we focus inner iframe contents if the webview itself is already focused
fix: oops on playmode
fix(i-p-flag): handle flag not associated to activity
fix for db.copy working in nativescript environment #<I>\n\nor any environment which is not properly environment detected
install map and set iterator shims
Easier to read debug output
Fix bug with detecting escaped quote
Supports es6 tests, resolves #<I>\n\nImproves error messages around parse errors (includes which file had a syntax\nerror)
expose html-parser example in elenpi at exports
Fix issue where plugin can resolve its promise before the gzip stream is done writing
Allow to apply migrations using migrationService not only for MongoDB because it doesn't have direct db dependencies\n\nFixes #<I>
Added interface for readers/writers
chore: check changelog script skip prerelease
(web admin) adjust description for associated sources\n\nrelated to #<I>
Adds assertion for misnamed controller name provided to render helper. fixes #<I>
minimumInputLength: from 2 to 1\nsearching on values with one 1 character
Docs: Fix missing pages  (#<I>)
Adds location hash to prevent <I> on sub-route github pages.
gdax: fetchOHLCV (): fix off-by-one error using limit arg.
[tcl mode] More liberally recognize comments\n\nCloses #<I>
named function for constructor rather than assigned variable to anonymous function
slightly fixed glitches in the latest Firefox
chore: simplify linked development by resolving from cwd
Bug fix: gulp-nightwatch always specify an 'output' directory as nightwatch's output location
Update Platform.js\n\nOptimise querying NPM registry.
rename private function thread to immediate
Fixed bug with conditional comments\n\ngetFiles function strips away all html comments before parsing the files. This does not have any specific utility and it has the unpleasant effect of removing all files wrapped inside a conditional comment, thus resulting in an empty file. Removing this behaviour.
apply exarkun's patch from #<I>.  closes #<I>
Add ability to use different configs (#<I>)
Return a vao - first render was failing
changed the cache to be based on documents
Do not show the explanation for the debugger if the debugger is not\nenabled.  This fixes #<I>.
Bugfix to clean up dead coroutines.
fixing repeat results on output (#9)
Add missing import at `index.js`\n\nfor registering version of addon for Ember Inspector
Improve detect to cover the incomplete implementation in IE8
feat: ability to cancel frameDebounce()
Got rid of some unused variables.
fix: pass string to web-push library\n\nAccording to the implementation of web-push (i.e. <URL>) the library expects a string instead of an object.
use "===" when testing for Date.now
Don't disable dojo-sync-loader in the loader extension config as it\nmight be needed by i<I>n::getLocalization()
Bring back protectVTree call, caught in component tests
merging parts of #<I>
Fix activity indicator for Ensembl-JSON sources.
Fix SQLite dropColumn doesn't work for last column (fixes #<I>)
[PLOT] Usually you want to initialize a variable before using it.
[htmlmixed mode] Remove obsolete property
